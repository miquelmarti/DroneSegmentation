#!/usr/bin/env python

"""Carries out transfer learning according to a provided configuration file."""

# auto-generated python class for protobuf formats
import transferLearning_pb2
import argparse
import os
import warnings
import logging

FILENAME_FIELD = 'filename'
NET_FILENAME_FIELD = 'net_filename'
HALT_FIELD = 'halt_percentage'
INIT_STAGE_FIELD = 'init_stage'


def getArguments():
    """Defines and parses command-line arguments to this script."""
    parser = argparse.ArgumentParser()
    # optional arguments
    parser.add_argument('--clean', action="store_true", help='\
    Cleans up intermediate files as the script finishes with them.')
    parser.add_argument('--weights', help='\
    A .caffemodel file containing the initial weights of the first stage.  \
    If not provided, the first stage will learn all weights from scratch.')
    machineGroup = parser.add_mutually_exclusive_group()
    machineGroup.add_argument('--cpu', action='store_true', help='\
    If this flag is set, runs all training on the CPU.')
    machineGroup.add_argument('--gpu', type=int, default=0, help='\
    Allows the user to specify which GPU training will run on.')
    parser.add_argument('--quiet', action='store_true', help='\
    Run in non-verbose mode.')
    parser.add_argument('-o', '--out_dir', help='\
    A directory in which to store the output caffe models.  Overrides \
    out_dir specified in configuration file, if any.')

    # required arguments
    parser.add_argument('config', help='\
    A .prototxt file defining the transfer learning stages to be performed.')
    return parser.parse_args()


def getStageFromMsg(stageMsg, configDir, outDir):
    # unpack values
    preProcFun = None
    if stageMsg.fcn_surgery:
        preProcFun = fcnSurgery.fcnInterp
    haltPercent = None
    if stageMsg.HasField(HALT_FIELD):
        haltPercent = stageMsg.halt_percentage
    # TODO make solver_filename relative to config file
    solverFilename = os.path.join(configDir, stageMsg.solver_filename)
    return stage.Stage(stageMsg.name, solverFilename, stageMsg.freeze,
                       stageMsg.ignore, preProcFun, haltPercent, outDir)
    

def getStagesFromMsgs(multiSourceMsg, configDir, outDir):
    """Instantiates a sequence of stages from protobuf "stage" messages."""
    stages = [getStageFromMsg(stageMsg, configDir, outDir)
              for stageMsg in multiSourceMsg.stage]
    initStage = None
    if multiSourceMsg.HasField(INIT_STAGE_FIELD):
        getStageFromMsg(multiSourceMsg.init_stage, configDir, outDir)
    return initStage, stages


def executeListOfStages(stages, firstModel, clean=False):
    model = firstModel
    allResults = []
    scores = None
    
    print 'Will execute the following stages :', [s.name for s in stages]
    
    for s in stages:
        print '-> Execute stage', s.name
        newModel, scores = s.execute(model)
        if clean:
            # delete each model as soon as we're finished with it
            if model != firstModel:
                os.remove(model)
        else:
            # save the nextResults of each model
            allResults.append((newModel, scores))
        model = newModel
        print '-> Produce the model', model

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
    if args.quiet:
        os.environ['GLOG_minloglevel'] = '3'
        warnings.filterwarnings("ignore")
        logging.basicConfig(level=logging.INFO)
        
    # Must change log level prior to importing caffe
    import caffe
    from caffeUtils import protoUtils, fcnSurgery
    import stage

    # Set the caffe device
    if args.cpu:
        caffe.set_mode_cpu()
    else:
        caffe.set_device(args.gpu)
        caffe.set_mode_gpu()

    # Read in the configuration file
    tlMsg = transferLearning_pb2.TransferLearning()
    protoUtils.readFromPrototxt(tlMsg, args.config)
    configDir = os.path.dirname(args.config)
    
    # Command-line out dir takes priority
    outDir = args.out_dir
    if outDir is None:
        # then config file out dir, relative to config file's location
        # TODO ; if tlMsg.out_dir, then get it as a full path,
        # else, get the dirname of args.stages
        outDir = os.path.join(configDir, tlMsg.out_dir)

    # Command-line init weights take priority
    prevModel = args.weights
    if prevModel is None:
        if os.path.isabs(tlMsg.init_weights):
            prevModel = tlMsg.init_weights
        else:
            prevModel = os.path.join(configDir, tlMsg.init_weights)

    # Execute all multi_source stage sequences in order
    prevModel = args.weights
    for msMsg in tlMsg.multi_source:
        if not msMsg.iterations > 0:
            raise Exception("The number of iterations should be > 0.")

        initStage, stages = getStagesFromMsgs(msMsg, configDir, outDir)
        if initStage is not None:
            pass  # TODO implement this!
        bestModels = [None] * len(stages)
        bestScores = [float('-inf')] * len(stages)

        for j in range(msMsg.iterations):
            print "starting iteration from", prevModel
            nextResults = None
            if initStage is not None and j is 0:
                # on the first iteration, run initStage instead of first stage
                firstStages = [initStage] + stages[1:]
                nextResults = executeListOfStages(firstStages, prevModel)
            else:
                nextResults = executeListOfStages(stages, prevModel)
            nextModels, nextScores = zip(*nextResults)

            # check if these are the new best models for their stage
            for i in range(len(bestModels)):
                if nextScores[i] is None:
                    print 'nextScores is none - assume next model is better'
                    # no scores given, so just assume it's better
                    bestModels[i] = nextModels[i]
                else:
                    # Save each stage's model if it's better than the previous
                    iNextScore = getScore(nextScores[i], msMsg.score_metric)
                    if iNextScore > bestScores[i]:
                        bestModels[i] = nextModels[i]
                        bestScores[i] = iNextScore
                        print ''.join(["New best score for stage ",
                                       stages[i].name, ": ", str(iNextScore)])
                    else:
                        print 'next score of', i, 'is better than previous'
                        
            # input model of the next iteration is the best last-stage model
            prevModel = nextModels[-1]
            
            # delete any models that are no longer needed
            for model in nextModels:
                shouldDelete = (model is not None and
                                model not in bestModels and
                                model != prevModel)
                if shouldDelete:
                    os.remove(model)
    
    print 'Final models stored in', ', '.join(bestModels)
    raise SystemExit
    
