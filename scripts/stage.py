import os
import caffe
from caffe.proto import caffe_pb2
import caffeUtils
import subprocess

LR_MULT_FIELD = 'lr_mult'
LAYER_PARAM_FIELD = 'param'
TEMP_FILE_SUFFIX = '.tmp'
IGNORE_LAYER_SUFFIX = '-ignore'
MODEL_SUFFIX = '.caffemodel'


def swapFiles(filename1, filename2):
    tempName = filename1 + TEMP_FILE_SUFFIX
    if tempName == filename2:
        tempName = tempName + TEMP_FILE_SUFFIX
    os.rename(filename1, tempName)
    os.rename(filename2, filename1)
    os.rename(tempName, filename2)


def freezeNetworkLayers(freezeList, trainNetFilename):
    trainNet = caffe_pb2.NetParameter()
    caffeUtils.readFromPrototxt(trainNet, trainNetFilename)

    for layer in trainNet.layer:
        # freeze desired layers
        if layer.name in freezeList:
            if layer.HasField(LAYER_PARAM_FIELD):
                for p in layer.param:
                    if p.HasField(LR_MULT_FIELD):
                        p.lr_mult = 0
            else:
                p = layer.param.add()
                p.lr_mult = 0

    # re-serialize modified files
    tmpTrainNetFilename = trainNetFilename + TEMP_FILE_SUFFIX
    caffeUtils.writeToPrototxt(trainNet, tmpTrainNetFilename)
    return tmpTrainNetFilename


def ignoreModelLayers(ignoreList, modelFilename):
    model = caffeUtils.readCaffeModel(modelFilename)
    ignoreNames = [layer.name for layer in ignoreList]
    for layer in model.layer:
        if layer.name in ignoreNames:
            # rename layers to force Caffe to ignore them
            layer.name = layer.name + IGNORE_LAYER_SUFFIX
    # write the model back out to a temporary file
    tmpModelFilename = modelFilename + TEMP_FILE_SUFFIX
    with open(tmpModelFilename, 'w') as f:
        f.write(model.SerializeToString())
    return tmpModelFilename

    
class Stage(object):
    """
    Constructor for the Stage class.

    Arguments:
    name -- The name of this stage
    freezeList -- a list of names of layers to be frozen while training
    ignoreList -- a list of names of layers in the model to be ignored
    trainNetFilename -- a prototxt file describing the training network
    """

    def __init__(self, name, freezeList, ignoreList, trainNetFilename=None):
        self.name = name
        self.freezeList = freezeList
        self.ignoreList = ignoreList
        self.trainNetFilename = trainNetFilename

    def execute(self, modelFilename):
        """Subclasses MUST override this method with some functionality."""
        return modelFilename


class PrototxtStage(Stage):
    """
    A class that represents a transfer-learning stage carried out based on a
    provided .prototxt file.

    Allows specification of provided weight layers to freeze and to ignore.
    """

    def __init__(self, name, solverFilename, freezeList, ignoreList,
                 trainNetFilename=None):
        """
        Constructor for the PrototxtStage class.

        Arguments:
        name -- The name of this stage
        solverFilename -- filename of a solver prototxt file for this stage
        freezeList -- a list of names of layers to be frozen while training
        ignoreList -- a list of names of layers in the model to be ignored
        trainNetFilename -- overrrides network referenced in the solver file
        """

        super(PrototxtStage, self).__init__(name, freezeList, ignoreList)
        self.solverFilename = solverFilename
        if trainNetFilename:
            self.trainNetFilename = trainNetFilename
        else:
            self.trainNetFilename = caffeUtils.getTrainNetFilename(
                solverFilename)

    def execute(self, modelFilename=None):
        """Executes this learning stage in caffe."""

        # read in the provided caffe config files
        tmpFilenames = []
        trainNetFilename = self.trainNetFilename
        solverFilename = self.solverFilename
        if modelFilename and len(self.freezeList) > 0:
            trainNetFilename = freezeNetworkLayers(self.freezeList,
                                                   trainNetFilename)
            solverSpec = caffeUtils.readSolver()
            solverSpec.net = trainNetFilename
            solverFilename = solverFilename + TEMP_FILE_SUFFIX
            caffeUtils.writeToPrototxt(solverSpec, solverFilename)
            tmpFilenames += [solverFilename, trainNetFilename]

        if modelFilename and len(self.ignoreList) > 0:
            modelFilename = ignoreModelLayers(self.ignoreList, modelFilename)
            tmpFilenames.append(modelFilename)

        # run caffe with the provided network description and solver info
        solver = caffe.get_solver(str(solverFilename))
        if modelFilename:
            solver.net.copy_from(modelFilename)
        freezeNames = [layer.name for layer in self.freezeList]
        sample = freezeNames[0]
        # TODO remove this test code
        frozenPre = solver.net.params[sample][0].data
        print "frozen before:", solver.net.params[sample][0].data
        solver.solve()
        frozenPost = solver.net.params[sample][0].data
        print "frozen after:", solver.net.params[sample][0].data

        outModelFilename = self.name + MODEL_SUFFIX
        solver.net.save(outModelFilename)
        # remove temporary files
        map(os.remove, tmpFilenames)
        return outModelFilename


class CommandStage(Stage):

    def __init__(self, name, command, outModelFilename, trainNetFilename=None,
                 freezeList=[], ignoreList=[]):
        """
        Constructor for the CommandStage class.

        Arguments:
        name -- The name of this stage
        command -- the command to be run to carry out the learning.
        outModelFilename -- the name of the .caffemodel file output by the
        solver command.
        trainNetFilename -- a prototxt file describing the training network
        freezeList -- a list of names of layers to be frozen while training.
        trainNetFilename must also be specified for freezeList to be applied.
        ignoreList -- a list of names of layers in the model to be ignored.
        """
        super(CommandStage, self).__init__(name, freezeList, ignoreList,
                                           trainNetFilename)
        self.command = command
        self.outModelFilename = outModelFilename

    def execute(self, modelFilename=None):
        trainNetFilename = self.trainNetFilename
        tmpTrainNetFilename = None
        tmpModelFilename = None
        if modelFilename and len(self.ignoreList) > 0:
            # apply ignore list
            tmpModelFilename = ignoreModelLayers(self.ignoreList,
                                                 modelFilename)
            swapFiles(modelFilename, tmpModelFilename)
        if modelFilename and trainNetFilename and len(self.freezeList) > 0:
            # apply freeze list
            tmpTrainNetFilename = freezeNetworkLayers(self.freezeList,
                                                      trainNetFilename)
            swapFiles(trainNetFilename, tmpTrainNetFilename)
        
        # execute the command
        retcode = subprocess.call(self.command)
        # restore the original network and caffemodel files
        if tmpModelFilename and modelFilename:
            os.rename(tmpModelFilename, modelFilename)
        if tmpTrainNetFilename and trainNetFilename:
            os.rename(tmpTrainNetFilename, trainNetFilename)

        if retcode is 0:
            return self.outModelFilename
        else:
            # The provided command exited abnormally!
            raise Exception("Solve command in stage " + self.name +
                            " returned code " + str(retcode))
