import os
import caffe
from caffe.proto import caffe_pb2
from caffeUtils import protoUtils, score, solve

# names of parameters in protobufs so we can check for their presence
LR_MULT_FIELD = 'lr_mult'
LAYER_PARAM_FIELD = 'param'

TEMP_FILE_SUFFIX = '.tmp'
IGNORE_LAYER_SUFFIX = '-ignore'
MODEL_SUFFIX = '.caffemodel'
SNAPSHOT_FORMAT_FIELD = 'snapshot_format'
ITER_PREFIX = '_iter_'

# keys for returned dictionary
OUT_MODEL_KEY = 'outModelFilename'
SCORE_KEY = 'score'


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
    """
    A class that represents a transfer-learning stage carried out based on a
    provided .prototxt file.

    Allows specification of provided weight layers to freeze and to ignore.
    """

    def __init__(self, name, solverFilename, freezeList, ignoreList,
                 preProcFun=None, haltPercentage=None):
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
        self.preProcFun = preProcFun
        self.haltPercentage = haltPercentage

    def cleanup(self, keepModelFilename):
        solverSpec = protoUtils.readSolver(str(self.solverFilename))
        filePrefix = os.path.basename(solverSpec.snapshot_prefix)
        # TODO is this correct?  Do we need to include the solverSpec dir?
        files = os.listdir(self.getSnapshotDir())
        for f in files:
            if f.startswith(filePrefix) and f != keepModelFilename:
                os.remove(f)

    def getSnapshotDir(self):
        solverDir = os.path.dirname(self.solverFilename)
        solverSpec = protoUtils.readSolver(str(self.solverFilename))
        relSnapshotDir, filePrefix = os.path.split(solverSpec.snapshot_prefix)
        snapshotDir = os.path.join(solverDir, relSnapshotDir)
        return snapshotDir

    def execute(self, modelFilename=None, usePySolver=True):
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

        # make sure that the output filename isn't already used
        outModelFilename = self.name + MODEL_SUFFIX
        i = 0
        while True:
            if not os.path.isfile(outModelFilename):
                break
            i += 1
            outModelFilename = '.'.join([outModelFilename, str(i)])

        # TODO allow user to specify loss layer, out layer, data layer, and
        # label layer names.
        scores = None
        if usePySolver:
            scores = solve.solve(solverFilename, modelFilename,
                                 outModelFilename, self.preProcFun,
                                 self.haltPercentage)
        else:
            solver = caffe.get_solver(str(solverFilename))
            if modelFilename:
                solver.net.copy_from(str(os.path.abspath(modelFilename)))
            solver.solve()
            solverSpec = protoUtils.readSolver(str(self.solverFilename))
            scores = solve.runValidation(solver, solverSpec.testIter[0],
                                         outLayer='score', lossLayer='loss',
                                         labelLayer='label')
            solver.net.save(str(outModelFilename))

        # remove temporary files
        self.cleanup(outModelFilename)
        map(os.remove, tmpFilenames)
        return {OUT_MODEL_KEY: outModelFilename, SCORE_KEY: scores}
