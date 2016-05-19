import os
import subprocess
import caffe
from caffe.proto import caffe_pb2
from caffeUtils import protoUtils

LR_MULT_FIELD = 'lr_mult'
LAYER_PARAM_FIELD = 'param'
TEMP_FILE_SUFFIX = '.tmp'
IGNORE_LAYER_SUFFIX = '-ignore'
DUPPLICATE_SUFFIX = '.nw'
MODEL_SUFFIX = '.caffemodel'
SNAPSHOT_FORMAT_FIELD = 'snapshot_format'
ITER_PREFIX = '_iter_'

TEST = True


def swapFiles(filename1, filename2):
    tempName = filename1 + TEMP_FILE_SUFFIX
    if tempName == filename2:
        tempName = tempName + TEMP_FILE_SUFFIX
    os.rename(filename1, tempName)
    os.rename(filename2, filename1)
    os.rename(tempName, filename2)


def freezeNetworkLayers(freezeList, trainNetFilename):
    trainNet = caffe_pb2.NetParameter()
    protoUtils.readFromPrototxt(trainNet, trainNetFilename)

    for layer in trainNet.layer:
        # freeze desired layers
        if layer.name in freezeList:
            if len(layer.param) > 0:
                for p in layer.param:
                    p.lr_mult = 0
            else:
                p = layer.param.add()
                p.lr_mult = 0

    # re-serialize modified files
    tmpTrainNetFilename = trainNetFilename + TEMP_FILE_SUFFIX
    protoUtils.writeToPrototxt(trainNet, tmpTrainNetFilename)
    return tmpTrainNetFilename


def ignoreModelLayers(ignoreList, modelFilename):
    model = protoUtils.readCaffeModel(modelFilename)
    layers = model.layer

    if len(layers) is 0:
        # model uses the deprecated V1LayerParameter, so handle these
        layers = model.layers

    # rename layers to force Caffe to ignore them
    for layer in layers:
        if layer.name in ignoreList:
            layer.name = layer.name + IGNORE_LAYER_SUFFIX

    # write the model back out to a temporary file
    tmpModelFilename = modelFilename + TEMP_FILE_SUFFIX
    with open(tmpModelFilename, 'w') as f:
        f.write(model.SerializeToString())
    return tmpModelFilename


class Stage(object):

    def __init__(self, name, solverFilename, freezeList, ignoreList):
        """
        Constructor for the Stage class.

        Arguments:
        name -- The name of this stage
        solverFilename -- filename of a solver prototxt file for this stage
        freezeList -- a list of names of layers to be frozen while training
        ignoreList -- a list of names of layers in the model to be ignored
        """

        self.name = name
        self.solverFilename = solverFilename
        self.freezeList = freezeList
        self.ignoreList = ignoreList
        self.trainNetFilename = protoUtils.getTrainNetFilename(solverFilename)

    def execute(self, modelFilename):
        """Subclasses MUST override this method with some functionality."""
        return modelFilename

    def cleanup(self, keepModelFilename):
        solverSpec = protoUtils.readSolver(str(self.solverFilename))
        filePrefix = os.path.basename(solverSpec.snapshot_prefix)
        # TODO is this correct?  Do we need to include the solverSpec dir?
        snapShotDir = self.getSnapshotDir()[0]
        files = os.listdir(snapShotDir)
        for f in files:
            if f.startswith(filePrefix) and os.path.join(snapShotDir, f) != keepModelFilename:
                print 'remove', f
                os.remove(os.path.join(snapShotDir, f))
        print 'keep', keepModelFilename
       
    def getSnapshotDir(self):
        solverSpec = protoUtils.readSolver(str(self.solverFilename))
        relSnapshotDir, filePrefix = os.path.split(solverSpec.snapshot_prefix)
        return relSnapshotDir, filePrefix


class PrototxtStage(Stage):
    """
    A class that represents a transfer-learning stage carried out based on a
    provided .prototxt file.

    Allows specification of provided weight layers to freeze and to ignore.
    """

    def __init__(self, name, solverFilename, freezeList, ignoreList):
        """
        Constructor for the PrototxtStage class.

        Arguments:
        name -- The name of this stage
        solverFilename -- filename of a solver prototxt file for this stage
        freezeList -- a list of names of layers to be frozen while training
        ignoreList -- a list of names of layers in the model to be ignored
        """

        super(PrototxtStage, self).__init__(name, solverFilename, freezeList,
                                            ignoreList)

    def execute(self, modelFilename=None):
        """Carries out this learning stage in caffe."""

        # read in the provided caffe config files
        tmpFilenames = []
        trainNetFilename = self.trainNetFilename
        solverFilename = self.solverFilename
        if modelFilename and len(self.freezeList) > 0:
            trainNetFilename = freezeNetworkLayers(self.freezeList,
                                                   trainNetFilename)
            solverSpec = protoUtils.readSolver(str(solverFilename))
            solverSpec.net = trainNetFilename
            solverFilename = solverFilename + TEMP_FILE_SUFFIX
            protoUtils.writeToPrototxt(solverSpec, solverFilename)
            tmpFilenames += [solverFilename, trainNetFilename]

        if modelFilename and len(self.ignoreList) > 0:
            modelFilename = ignoreModelLayers(self.ignoreList, modelFilename)
            tmpFilenames.append(modelFilename)

        # run caffe with the provided network description and solver info
        solver = caffe.get_solver(str(solverFilename))
        if modelFilename:
            solver.net.copy_from(str(os.path.abspath(modelFilename)))
        solver.solve()
        outModelFilename = self.name + MODEL_SUFFIX
        if os.path.isfile(outModelFilename):
            outModelFilename += DUPPLICATE_SUFFIX
        solver.net.save(str(outModelFilename))
        # remove temporary files
        map(os.remove, tmpFilenames)
        return outModelFilename


class CommandStage(Stage):

    def __init__(self, name, command, solverFilename, outModelFilename=None,
                 freezeList=[], ignoreList=[]):
        """
        Constructor for the CommandStage class.

        Arguments:
        name -- The name of this stage
        command -- the command to be run to carry out the learning.
        outModelFilename -- the name of the .caffemodel file output by the
        solver command.
        freezeList -- a list of names of layers to be frozen while training.
        trainNetFilename must also be specified for freezeList to be applied.
        ignoreList -- a list of names of layers in the model to be ignored.
        """
        super(CommandStage, self).__init__(name, solverFilename, freezeList,
                                           ignoreList)
        self.command = command
        self.outModelFilename = outModelFilename

    def execute(self, modelFilename=None):
        trainNetFilename = self.trainNetFilename
        tmpTrainNetFilename = None
        tmpModelFilename = None
        outModelFilename = self.outModelFilename
        if modelFilename and len(self.ignoreList) > 0:
            tmpModelFilename = ignoreModelLayers(self.ignoreList,
                                                 modelFilename)
            swapFiles(modelFilename, tmpModelFilename)
        if modelFilename and trainNetFilename and len(self.freezeList) > 0:
            # apply freeze list
            tmpTrainNetFilename = freezeNetworkLayers(self.freezeList,
                                                      trainNetFilename)
            swapFiles(trainNetFilename, tmpTrainNetFilename)

        # execute the command
        try:
            retcode = subprocess.call(self.command, shell=True)
        finally:
            # restore the original network and caffemodel files
            if tmpModelFilename and modelFilename:
                os.rename(tmpModelFilename, modelFilename)
            if tmpTrainNetFilename and trainNetFilename:
                os.rename(tmpTrainNetFilename, trainNetFilename)
            # TODO if this model isn the best one, the previous one (from the previous iteration in case of multi source) is erased... not good. Need to systematically same a copy of these weights (if they are the best) in a given path
            if outModelFilename is None and retcode is 0:
                outModelFilename = self.getLastIterModel()
            self.cleanup(outModelFilename)

        if retcode is 0:
            return outModelFilename
        else:
            # The provided command exited abnormally!
            raise Exception("Solve command in stage " + self.name +
                            " returned code " + str(retcode))

    def getLastIterModel(self):
        snapshotDir, snapshotPrefix = self.getSnapshotDir()
        lastModel = None
        maxIterNum = 0
        # TODO some folders will have snapshots from different trainings, like cifar_iter_.. and cifar_lr1_iter_..., deal with them, taking the highest one is not enough
        print self.getSnapshotDir()
        for filename in os.listdir(snapshotDir):
            begin, middle, end = filename.rpartition(ITER_PREFIX)
            if begin == snapshotPrefix and middle == ITER_PREFIX and MODEL_SUFFIX in end:
                iterNum = int(end.partition('.')[0])
                if iterNum > maxIterNum:
                    maxIterNum = iterNum
                    lastModel = filename
        print 'snapshotDir', snapshotDir
        print 'lastModel', lastModel
        print 'list', os.listdir(snapshotDir)
        return os.path.join(snapshotDir, lastModel)
