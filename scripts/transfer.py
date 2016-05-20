#!/usr/bin/env python

"""Carries out transfer learning according to a provided configuration file."""

# auto-generated python class for protobuf formats
import transferLearning_pb2
import caffe
import argparse
import os
from caffeUtils import protoUtils, fcnSurgery
import stage

FILENAME_FIELD = 'filename'
NET_FILENAME_FIELD = 'net_filename'
HALT_FIELD = 'halt_percentage'


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
    parser.add_argument('--quiet', action='store_true', help='\
    Non-verbose version.')

    # required arguments
    parser.add_argument('stages', help='\
    A .prototxt file defining the transfer learning stages to be performed.')
    return parser.parse_args()


def getStagesFromMsgs(stageMsgs, solverFilename=None):
    """Instantiates a sequence of stages from protobuf "stage" messages."""
    stages = []
    for stageMsg in stageMsgs:
        # unpack values
        preProcFun = None
        if stageMsg.fcn_surgery:
            preProcFun = fcnSurgery.fcnInterp
        haltPercent = None
        if stageMsg.HasField(HALT_FIELD):
            haltPercent = stageMsg.halt_percentage
        # add a new stage to the list
        newStage = stage.Stage(stageMsg.name, stageMsg.solver_filename,
                               stageMsg.freeze, stageMsg.ignore,
                               preProcFun, haltPercent)
        stages.append(newStage)
    return stages


def executeListOfStages(stages, firstModel, clean=False):
    model = firstModel
    allResults = []
    scores = None
    for s in stages:
        newModel, scores = s.execute(model)
        if clean:
            # delete each model as soon as we're finished with it
            if model != firstModel:
                os.remove(model)
        else:
            # save the nextResults of each model
            allResults.append((newModel, scores))
        model = newModel

    if clean:
        # only the last was saved
        allResults = [(model, scores)]
    return allResults


def getScore(scores, scoreMetric):
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
    if args.quiet:
        os.environ['GLOG_minloglevel'] = '3'
    
    # Read in the stages and carry them out
    tlMsg = transferLearning_pb2.TransferLearning()
    protoUtils.readFromPrototxt(tlMsg, args.stages)
    stages = getStagesFromMsgs(tlMsg.stage)
    model, scores = executeListOfStages(stages, args.model, args.clean)[-1]
    if len(tlMsg.multi_source) is 0:
        # We're done!
        print 'Final model stored in', model
        exit(0)
        
    # Run all multi-source stage sequences
    prevModel = model
    for ms in tlMsg.multi_source:
        stages = getStagesFromMsgs(ms.stage)
        bestModels = [None] * len(stages)
        bestScores = [float('-inf')] * len(stages)

        for _ in range(ms.iterations):
            nextResults = executeListOfStages(stages, prevModel)
            nextModels, nextScores = zip(*nextResults)
            # check if these are the new best models for their stage
            for i in range(len(bestModels)):
                iNextScore = getScore(nextScores[i], ms.score_metric)
                if iNextScore > bestScores[i]:
                    bestModels[i] = nextModels[i]
                    bestScores[i] = iNextScore
                    msg = ''.join(["New best score for stage ", stages[i].name,
                                   ": ", str(iNextScore)])
                    print msg

            # previous model is no longer needed (unless it's the best one)
            if prevModel is not None and prevModel not in bestModels:
                os.remove(prevModel)
            # input model of the next iteration is output of final stage
            prevModel = nextModels[-1]

            # clean up any remaining unneeded models
            for model in nextModels:
                if model is not None and model not in bestModels:
                    os.remove(model)

    print 'Final models stored in', ', '.join(bestModels)
    exit(0)
