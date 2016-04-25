import os
import caffe
from caffe.proto import caffe_pb2
import caffeUtils

LR_MULT_FIELD = 'lr_mult'
LAYER_PARAM_FIELD = 'param'
TEMP_FILE_SUFFIX = '.tmp'
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
        ignoreList -- a list of names of layers in the weight file to be ignored
        trainNetFilename -- overrrides any net model included in the solver file
        """
        self.name = name
        self.solverFilename = solverFilename
        self.freezeList = freezeList
        self.ignoreList = ignoreList
        if trainNetFilename:
            self.trainNetFilename = trainNetFilename
        else:
            self.trainNetFilename = caffeUtils.getTrainNetFilename(solverFilename)

                
    def execute(self, model=None):
        """Executes this learning stage in caffe."""
        
        # read in the provided caffe config files
        tmpFilenames = []
        trainNetFilename = self.trainNetFilename
        solverFilename = self.solverFilename
        if model:
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

                if layer.name in self.ignoreList:
                    # TODO how to do this? randomly rename the "restart" layers?
                    pass

            # re-serialize modified files
            trainNetFilename = trainNetFilename + TEMP_FILE_SUFFIX
            caffeUtils.writeToPrototxt(trainNet, trainNetFilename)
            solverSpec = caffeUtils.readSolver()
            solverSpec.net = trainNetFilename
            solverFilename = solverFilename + TEMP_FILE_SUFFIX
            caffeUtils.writeToPrototxt(solverSpec, solverFilename)
            tmpFilenames += [solverFilename, trainNetFilename]

        # run caffe with the provided network description and solver info 
        solver = caffe.get_solver(str(solverFilename))
        if model:
            solver.net.copy_from(model)
        solver.solve()

        outModelFilename = self.name + MODEL_SUFFIX
        solver.net.save(outModelFilename)
        # remove temporary files
        map(os.remove, tmpFilenames)
        return outModelFilename

