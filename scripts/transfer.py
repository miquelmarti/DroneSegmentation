#!/usr/bin/env python

"""Carries out transfer learning according to a provided configuration file."""

# auto-generated python class for protobuf formats
import transferLearning_pb2
import caffe
import argparse
import os
from caffeUtils import protoUtils, score, iterators
from stage import PrototxtStage, CommandStage
import numpy as np

FILENAME_FIELD = 'filename'
NET_FILENAME_FIELD = 'net_filename'
SOLVE_CMD_FIELD = 'cmd_solver'
OUT_MODEL_FILENAME_FIELD = 'out_model_filename'


def getArguments():
    """Defines and parses command-line arguments to this script."""
    parser = argparse.ArgumentParser()
    # optional arguments
    parser.add_argument('--clean', action="store_true", help='\
    Cleans up intermediate files as the script finishes with them.')
    parser.add_argument('--model', help='\
    A .caffemodel file containing the initial weights of the first stage.  \
    If not provided, the first stage will learn all weights from scratch.')
    machineGroup = parser.add_mutually_exclusive_group()
    machineGroup.add_argument('--cpu', action='store_true', help='\
    If this flag is set, runs all training on the CPU.')
    machineGroup.add_argument('--gpu', type=int, default=0, help='\
    Allows the user to specify which GPU training will run on.')

    # required arguments
    parser.add_argument('stages', help='\
    A .prototxt file defining the transfer learning stages to be performed.')
    return parser.parse_args()


def getStagesFromMsgs(stageMsgs, solverFilename=None):
    """Instantiates a sequence of stages from protobuf "stage" messages."""
    stages = []
    for stageMsg in stageMsgs:
        # unpack values
        newStage = None
        solverFilename = stageMsg.solver_filename
        if not stageMsg.HasField(SOLVE_CMD_FIELD):
            newStage = PrototxtStage(stageMsg.name, solverFilename,
                                     stageMsg.freeze, stageMsg.ignore)
        else:
            outFilename = None
            if stageMsg.cmd_solver.HasField(OUT_MODEL_FILENAME_FIELD):
                outFilename = stageMsg.cmd_solver.out_model_filename
            
            newStage = CommandStage(stageMsg.name, stageMsg.cmd_solver.command,
                                    solverFilename, outFilename,
                                    freezeList=stageMsg.freeze,
                                    ignoreList=stageMsg.ignore)
            
        stages.append(newStage)
    return stages


def executeListOfStages(stages, firstModel, clean):
    model = firstModel
    for stage in stages:
        newModel = stage.execute(model)
        if clean and model != firstModel:
            os.remove(model)
        model = newModel
    return model


def computeScore(deployFilename, model, valSet, mean, scoreMetric,
                 outLayer='output'):
    scores = score.scoreDataset(deployFilename, model, valSet, mean, outLayer)
    if scoreMetric == transferLearning_pb2.MultiSource.MEAN_IU:
        return scores.meanIu
    elif scoreMetric == transferLearning_pb2.MultiSource.ACCURACY:
        return scores.meanAcc
    elif scoreMetric == transferLearning_pb2.MultiSource.LOSS:
        return scores.loss
    else:
        raise Exception("An invalid scoreMetric was specified: " +
                        str(scoreMetric))


if __name__ == "__main__":
    args = getArguments()
    if args.cpu:
        caffe.set_mode_cpu()
    else:
        caffe.set_device(args.gpu)
        caffe.set_mode_gpu()
    bestModel = args.model

    # Read in the stages and carry them out
    tlMsg = transferLearning_pb2.TransferLearning()
    protoUtils.readFromPrototxt(tlMsg, args.stages)
    stages = getStagesFromMsgs(tlMsg.stage)
    bestModel = executeListOfStages(stages, args.model, args.clean)

    if len(tlMsg.multi_source) > 0:
        # Run all multi-source stage sequences
        for ms in tlMsg.multi_source:
            prevModel = bestModel
            nextModel = None
            # TODO implement handling mean_file
            mean = None
            if ms.mean_value:
                mean = np.array(ms.mean_value)
            valSet = iterators.FileListIterator(ms.validation_set)
            bestScore = 0
            if bestModel is not None:
                bestScore = computeScore(ms.deploy_net, bestModel, valSet,
                                         mean, scoreMetric=ms.score_metric)
            print "New best model's score:", bestScore
            stages = getStagesFromMsgs(ms.stage)
            for i in range(ms.iterations):
                # learn the next stage in the sequence
                nextModel = executeListOfStages(stages, prevModel, args.clean)
                nextScore = computeScore(ms.deploy_net, nextModel,
                                         ms.validation_set, mean,
                                         scoreMetric=ms.score_metric)
                
                # check if this is the new best model
                if nextScore > bestScore:
                    bestModel = nextModel
                    bestScore = nextScore
                    print "New best model's score:", bestScore
                    
                # previous model is no longer needed (unless it's the best one)
                if prevModel != bestModel:
                    os.remove(prevModel)
                prevModel = nextModel
            # clean up any remaining unneeded model
            if nextModel is not None and nextModel != bestModel:
                os.remove(nextModel)

                
    print 'Final model stored in', bestModel
