# TODO: Check ignore and freeze functions

import os
import caffe
from caffe.proto import caffe_pb2
from caffeUtils import protoUtils, solve

# Suffixes
S_IGNORE_LAYER = '-ignore'

# Extensions
E_TEMP_FILE = '.tmp'
E_MODEL = '.caffemodel'

# General
ITER = '_iter_'


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
    tmpTrainNetFilename = trainNetFilename + E_TEMP_FILE
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
            layer.name = layer.name + S_IGNORE_LAYER

    # write the model back out to a temporary file
    tmpModelFilename = modelFilename + E_TEMP_FILE
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
        self.layerNames = (dataLayer, lossLayer, outLayer, labelLayer)
        
        learnDir = os.path.dirname(solverFilename)
        trainNetBasename = protoUtils.getTrainNetFilename(solverFilename)
        self.trainNetFilename = os.path.join(learnDir, trainNetBasename)
        self.outModelFilename = os.path.join(outDir, self.name + E_MODEL)

    def verify(self):
        """Check if the stage respects few requirements."""
        assert os.path.isfile(self.solverFilename), \
        ' '.join["Cannot find solver file:", solverFilename]
        if self.haltPercentage:
            assert self.haltPercentage >= 0 and self.haltPercentage <= 100, \
            ' '.join["'haltPercentage' should be set between 0 and 100, here:",
                     self.iteration]

    def cleanup(self, keepModelFilenames):
        """Cleans all the snapshots / models, except those we want to keep."""
        snapshotDir, snapshotPrefixBase = self.getSnapshotInfo()
        snapshotPrefix = snapshotPrefixBase + ITER
        for f in os.listdir(snapshotDir):
            fullFilePath = os.path.join(snapshotDir, f)
            doRemoveFile = (os.path.isfile(fullFilePath) and
                            f.startswith(snapshotPrefix) and
                            not f in keepModelFilenames)
            if doRemoveFile:
                os.remove(fullFilePath)
    
    def getSnapshotInfo(self):
        """Returns the directory where are the snapshots and their prefixes."""
        solverSpec = protoUtils.readSolver(str(self.solverFilename))
        solverDir = os.path.dirname(self.solverFilename)
        solverRelSnapshotDir, filePrefix = os.path.split(
            solverSpec.snapshot_prefix)
        snapshotDir = os.path.join(solverDir, solverRelSnapshotDir)
        return snapshotDir, filePrefix
    
    def execute(self, modelFilename=None, snapshot=None,
                      snapshotToRestore=None, quiet=False):
        """Carries out this learning stage in caffe."""
        # Will store all the temporary created files (to delete them after)
        tmpFilenames = []
        
        # Get solver main values
        solverFilename = self.solverFilename
        trainNetFilename = self.trainNetFilename
        outModelFilename = self.outModelFilename
        
        # Make sure that the output filename isn't already used, and change its
        # extension if it is (.1, .2, .3, ...)
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
            solverFilename = solverFilename + E_TEMP_FILE
            protoUtils.writeToPrototxt(solverSpec, solverFilename)
            tmpFilenames += [solverFilename, trainNetFilename]
        
        # Ignore some layers of the given model if needed
        if modelFilename and len(self.ignoreList) > 0:
            modelFilename = ignoreModelLayers(self.ignoreList, modelFilename)
            tmpFilenames.append(modelFilename)
        
        # Run the stage
        outNet, scores = solve.solve(solverFilename, modelFilename,
                                     self.preProcFun, self.haltPercentage,
                                     self.layerNames, snapshot, 
                                     snapshotToRestore, quiet)
        
        # Save the results
        outNet.save(str(outModelFilename))
        modelsToKeep = [outModelFilename]
        if snapshot.stage_snapshot:
            modelsToKeep.append(os.path.basename(snapshot.stage_snapshot))

        # Remove temporary files
        self.cleanup(modelsToKeep)
        map(os.remove, tmpFilenames)
        return outModelFilename, scores


