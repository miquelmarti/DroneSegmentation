#!/usr/bin/env python

"""Carries out transfer learning according to a provided configuration file."""

# auto-generated python classes for protobuf formats
import transferLearning_pb2
import caffe
from caffe.proto import caffe_pb2
import google.protobuf
import argparse
import os
import caffeUtils
from stage import Stage


SOLVER_FIELD = 'solver'


def getArguments():
    """Defines and parses command-line arguments to this script."""
    parser = argparse.ArgumentParser()
    # optional arguments
    parser.add_argument('--clean', action="store_true", help='\
    Cleans up intermediate .prototxt and .caffemodel files as the script \
    finishes with them.') 
    parser.add_argument('--weights', help='\
    A .caffemodel file containing the initial weights of the first stage.  \
    If not provided, the first stage will learn all weights from scratch.')

    machineGroup = parser.add_mutually_exclusive_group()
    machineGroup.add_argument('--cpu', action='store_true', help='\
    If this flag is set, runs all training on the CPU.')
    machineGroup.add_argument('--gpu', type=int, default=0, help='\
    Allows the user to specify which GPU training will run on.')

    # required arguments
    parser.add_argument('stages', help='\
    A .prototxt file defining the stages of transfer learning to be performed.')
    return parser.parse_args()


def getStagesFromMsgs(stageMsgs, solverFilename=None, modelFilename=None):
    """Instantiates a sequence of stages from protobuf "stage" messages."""
    stages = []
    for stageMsg in stageMsgs:
        # unpack values
        if stage.HasField(SOLVER_FIELD):
            solverFilename = stage.solver
        elif solverFilename is not None:
            stage.solver = solverFilename
        else:
            raise Exception('First layer provides no solver.prototxt file')
        # execute stage
        # TODO implement modelFilename logic
        stages.append(Stage(stageMsg.name, solverFilename, stageMsg.freeze,
                            stageMsg.ignore))
    return stages
    

def executeListOfStages(stages, firstWeightFile, clean):
    weights = firstWeightFile
    for stage in stages:
        newWeights = stage.execute(weights)
        if clean and weights != firstWeightFile:
            os.remove(weights)
        weights = newWeights
    return weights

    
if __name__ == "__main__":
    args = getArguments()
    if args.cpu:
        caffe.set_mode_cpu()
    else:
        caffe.set_device(args.gpu)
        caffe.set_mode_gpu()

    # Read in the stages and carry them out
    stageMsgs = transferLearning_pb2.TransferLearning()
    caffeUtils.readFromPrototxt(stages, args.stages)
    print "There are", len(stages), "stages specified"
    stages = getStagesFromMsgs(stageMsgs)
    executeListOfStages(stages, args.weights, args.clean)

