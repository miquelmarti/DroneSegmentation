import os
import caffe
from caffe.proto import caffe_pb2
import caffeUtils

LR_MULT_FIELD = 'lr_mult'
LAYER_PARAM_FIELD = 'param'
TEMP_FILE_SUFFIX = '.tmp'
IGNORE_LAYER_SUFFIX = '-ignore'
MODEL_SUFFIX = '.caffemodel'


class Stage(object):
    """
    A class that represents a transfer-learning stage.

    Allows specification of provided weight layers to freeze and to ignore.
    """

    def __init__(self, name, solverFilename, freezeList, ignoreList,
                 trainNetFilename=None):
        """
        Constructor for the Stage class.

        Arguments:
        name -- The name of this stage
        solverFilename -- filename of a solver prototxt file for this stage
        freezeList -- a list of names of layers to be frozen while training
        ignoreList -- a list of names of layers in the model to be ignored
        trainNetFilename -- overrrides network referred to in the solver file
        """
        self.name = name
        self.solverFilename = solverFilename
        self.freezeList = freezeList
        self.ignoreList = ignoreList
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
        if modelFilename:
            # load requisite variables
            trainNet = caffe_pb2.NetParameter()
            caffeUtils.readFromPrototxt(trainNet, self.trainNetFilename)

            for layer in trainNet.layer:
                # freeze desired layers
                if layer.name in self.freezeList:
                    if layer.HasField(LAYER_PARAM_FIELD):
                        for p in layer.param:
                            if p.HasField(LR_MULT_FIELD):
                                p.lr_mult = 0
                    else:
                        p = layer.param.add()
                        p.lr_mult = 0

            # re-serialize modified files
            trainNetFilename = trainNetFilename + TEMP_FILE_SUFFIX
            caffeUtils.writeToPrototxt(trainNet, trainNetFilename)
            solverSpec = caffeUtils.readSolver()
            solverSpec.net = trainNetFilename
            solverFilename = solverFilename + TEMP_FILE_SUFFIX
            caffeUtils.writeToPrototxt(solverSpec, solverFilename)
            tmpFilenames += [solverFilename, trainNetFilename]

        if modelFilename and len(self.ignoreList) > 0:
            model = caffeUtils.readCaffeModel(modelFilename)
            ignoreNames = [layer.name for layer in self.ignoreList]
            for layer in model.layer:
                if layer.name in ignoreNames:
                    # rename layers to force Caffe to ignore them
                    layer.name = layer.name + IGNORE_LAYER_SUFFIX
            # write the model back out to a temporary file
            tmpModelFilename = modelFilename + TEMP_FILE_SUFFIX
            with open(tmpModelFilename, 'w') as f:
                f.write(model.SerializeToString())
            modelFilename = tmpModelFilename
            tmpFilenames.append(modelFilename)

        # run caffe with the provided network description and solver info
        solver = caffe.get_solver(str(solverFilename))
        if modelFilename:
            solver.net.copy_from(modelFilename)
        freezeNames = [layer.name for layer in self.freezeList]
        sample = freezeNames[0]
        print "frozen before:", solver.net.params[sample][0].data
        solver.solve()
        print "frozen after:", solver.net.params[sample][0].data

        outModelFilename = self.name + MODEL_SUFFIX
        solver.net.save(outModelFilename)
        # remove temporary files
        map(os.remove, tmpFilenames)
        return outModelFilename
