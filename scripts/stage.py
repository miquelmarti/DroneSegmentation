# TODO: Check ignore and freeze functions
# TODO: Check solve.solve
# TODO: Make sure that usePySolver is useless

import os
import caffe
from caffe.proto import caffe_pb2
from caffeUtils import protoUtils, solve

# names of parameters in protobufs so we can check for their presence
LR_MULT_FIELD = 'lr_mult'
LAYER_PARAM_FIELD = 'param'
SNAPSHOT_FORMAT_FIELD = 'snapshot_format'

TEMP_FILE_SUFFIX = '.tmp'
IGNORE_LAYER_SUFFIX = '-ignore'
MODEL_SUFFIX = '.caffemodel'
ITER_PREFIX = '_iter_'


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
                 preProcFun=None, haltPercentage=None, outDir='',
                 dataLayer='data', lossLayer='loss', outLayer='out',
                 labelLayer='label'):
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
        self.preProcFun = preProcFun
        self.haltPercentage = haltPercentage
        self.dataLayer = dataLayer
        self.lossLayer = lossLayer
        self.outLayer = outLayer
        self.labelLayer = labelLayer
        
        learnDir = os.path.dirname(solverFilename)
        trainNetBasename = protoUtils.getTrainNetFilename(solverFilename)
        self.trainNetFilename = os.path.join(learnDir, trainNetBasename)
        self.outModelFilename = os.path.join(outDir, self.name + MODEL_SUFFIX)

    def cleanup(self, keepModelFilename):
        snapshotDir, snapshotPrefixBase = self.getSnapshotInfo()
        snapshotPrefix = snapshotPrefixBase + ITER_PREFIX
        for f in os.listdir(snapshotDir):
            fullFilePath = os.path.join(snapshotDir, f)
            doRemoveFile = (os.path.isfile(fullFilePath) and
                            f.startswith(snapshotPrefix) and
                            f != keepModelFilename)
            if doRemoveFile:
                os.remove(fullFilePath)
    
    def getSnapshotInfo(self):
        solverSpec = protoUtils.readSolver(str(self.solverFilename))
        solverDir = os.path.dirname(self.solverFilename)
        solverRelSnapshotDir, filePrefix = os.path.split(
            solverSpec.snapshot_prefix)
        snapshotDir = os.path.join(solverDir, solverRelSnapshotDir)
        return snapshotDir, filePrefix
    
    def execute(self, modelFilename=None):
        """Carries out this learning stage in caffe."""
        
        # Will store all the temporary created files (to delete them after)
        tmpFilenames = []
        
        solverFilename = self.solverFilename
        trainNetFilename = self.trainNetFilename
        outModelFilename = self.outModelFilename
        
        # Make sure that the output filename isn't already used
        i = 0
        while True:
            if not os.path.isfile(outModelFilename):
                break
            i += 1
            outModelFilename = '.'.join([self.outModelFilename, str(i)])
        
        # Freeze some layers of trainNet if needed
        if len(self.freezeList) > 0:
            trainNetFilename = freezeNetworkLayers(self.freezeList,
                                                   trainNetFilename)
            solverSpec = protoUtils.readSolver(str(solverFilename))
            # TODO: check if the net / train_net matters
            solverSpec = protoUtils.replaceTrainNetFilename(solverSpec, 
                                                            trainNetFilename)
            solverFilename = solverFilename + TEMP_FILE_SUFFIX
            protoUtils.writeToPrototxt(solverSpec, solverFilename)
            tmpFilenames += [solverFilename, trainNetFilename]
        
        # Ignore some layers of the given model if needed
        if modelFilename and len(self.ignoreList) > 0:
            modelFilename = ignoreModelLayers(self.ignoreList, modelFilename)
            tmpFilenames.append(modelFilename)
        
        outNet, scores = solve.solve(solverFilename, modelFilename,
                                     self.preProcFun, self.haltPercentage,
                                     self.dataLayer, self.lossLayer,
                                     self.outLayer, self.labelLayer)
        
        # Save the results
        outNet.save(str(outModelFilename))

        # Remove temporary files
        self.cleanup(outModelFilename)
        map(os.remove, tmpFilenames)
        return outModelFilename, scores


