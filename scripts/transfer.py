#!/usr/bin/env python
# TODO: Use logging
# TODO: Also take in charge absolute paths
# TODO: Think about the preProcFun -> should be usable easily

"""Carries out transfer learning according to a provided configuration file."""

# Auto-generated python class for protobuf formats
import transferLearning_pb2
import argparse
import os
import warnings
import logging

# Field names
F_HALT          = 'halt_percentage'
F_INIT_STAGE    = 'init_stage'
F_INIT_WEIGHTS  = 'init_weights'
F_OUT_DIR       = 'out_dir'

# Suffixes
S_SNAP          = '_snapshot'

# Extensions
E_PROTOTXT      = '.prototxt'



def getArguments():
    """Defines and parses command-line arguments to this script."""
    parser = argparse.ArgumentParser()
    
    # Optional arguments
    parser.add_argument('--weights', help='\
    A .caffemodel file containing the initial weights of the first stage. \
    Overrides init_weights specified in configuration file, if any. If not \
    provided, the first stage will learn all weights from scratch.')
    parser.add_argument('-o', '--out_dir', help='\
    A directory in which to store the output caffe models. Overrides out_dir \
    specified in configuration file, if any.')
    
    machineGroup = parser.add_mutually_exclusive_group()
    machineGroup.add_argument('--cpu', action='store_true', help='\
    If this flag is set, runs all training on the CPU.')
    machineGroup.add_argument('--gpu', type=int, default=0, help='\
    Allows the user to specify which GPU training will run on.')
    
    parser.add_argument('--clean', action="store_true", help='\
    Cleans up intermediate files as the script finishes with them.')
    parser.add_argument('--quiet', action='store_true', help='\
    Run in non-verbose mode.')
    parser.add_argument('--resume', help='\
    Use a snapshot file (generated by the framework but also modifiable) and \
    resume a training with it.')

    # Required arguments
    parser.add_argument('config', help='\
    A .prototxt file defining the transfer learning stages to be performed.')
    return parser.parse_args()


def getStageFromMsg(stageMsg, configDir, outDir):
    """Extract a stage from its message (prototxt format)."""
    # Unpack values
    preProcFun = None
    if stageMsg.fcn_surgery:
        preProcFun = fcnSurgery.fcnInterp
    haltPercent = None
    if stageMsg.HasField(F_HALT):
        haltPercent = stageMsg.halt_percentage
    
    # The solver_filename is relative to config file
    solverFilename = os.path.join(configDir, stageMsg.solver_filename)
    s = stage.Stage(stageMsg.name, solverFilename, stageMsg.freeze,
                    stageMsg.ignore, preProcFun, haltPercent, outDir,
                    stageMsg.dataLayer, stageMsg.lossLayer, stageMsg.outLayer,
                    stageMsg.labelLayer)
    
    # Check if the stage is ok
    s.verify()
    
    return s
    

def getStagesFromMsgs(multiSourceMsg, configDir, outDir):
    """Instantiates a sequence of stages from protobuf "stage" messages."""
    stages = [getStageFromMsg(stageMsg, configDir, outDir)
              for stageMsg in multiSourceMsg.stage]
    initStage = None
    if multiSourceMsg.HasField(F_INIT_STAGE):
        initStage = getStageFromMsg(multiSourceMsg.init_stage, configDir,
                                    outDir)
    return initStage, stages


def executeListOfStages(stages, firstModel, snapshot, snapshotToRestore=None,
                        clean=False, quiet=False):
    """Execute the sequence of stages."""
    model = firstModel
    allResults = []
    scores = None
    
    # If we have to resume a snapshot
    doRestore = snapshotToRestore and not snapshotToRestore.restored
    
    # Restore the weights for the stage to execute next, if provided
    if doRestore and snapshotToRestore.stage_weights \
                 and not snapshotToRestore.stage_weights[0] is '':
        model = snapshotToRestore.stage_weights
    
    # For each stage
    for idx, s in enumerate(stages):
        # Check if this stage has already been done
        if doRestore and snapshotToRestore.stage > idx:
            if not clean:
                allResults.append((None, None))
            print '>> Restored the stage', s.name
            continue
        
        # Update current snapshot
        setattr(snapshot, 'stage', idx)
        setattr(snapshot, 'stage_weights', model)
        print ' '.join(['>> Will start the stage', s.name, 
                        str(os.path.basename(model) if model else '')])
        
        # Save the snapshot
        snapshot.save()
        
        # Execute the stage
        newModel, scores = s.execute(model, snapshot, snapshotToRestore, quiet)
    
        # If we had to restore, set it as done here
        if snapshotToRestore:
            setattr(snapshotToRestore, 'restored', True)
            doRestore = False
        
        # Save or delete intermediate weights
        if clean:
            if model != firstModel:
                os.remove(model)
        else:
            allResults.append((newModel, scores))
        
        model = newModel
        print '>> Stage', s.name, 'produces model', os.path.basename(model)
    
    if clean:
        # Only the last was saved
        allResults = [(model, scores)]
    
    # Return the models produced. Warning, the shape is not the same depending
    # on the clean argument
    return allResults


def getScore(scores, scoreMetric):
    """Returns the appropriate metrics (defined in the config file)."""
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
    # Get the arguments
    args = getArguments()
    
    # Non-verbose mode
    if args.quiet:
        os.environ['GLOG_minloglevel'] = '3'
        warnings.filterwarnings("ignore")
        logging.basicConfig(level=logging.INFO)
        # TODO: delete protobuf warnings
    
    # Must change log level prior to importing caffe
    import caffe
    from caffeUtils import protoUtils, fcnSurgery
    import stage, snapshot

    # Set the caffe device
    if args.cpu:
        caffe.set_mode_cpu()
    else:
        # TODO: set_device(1) runs the framework on gpu 0, fix this
        caffe.set_device(args.gpu)
        caffe.set_mode_gpu()

    # Read in the configuration file
    tlMsg = transferLearning_pb2.TransferLearning()
    protoUtils.readFromPrototxt(tlMsg, args.config)
    configDir = os.path.dirname(args.config)
    
    # Initialize the snapshot to save
    snapshot = snapshot.Snapshot()
    
    # If we have to restore something
    snapshotToRestore = None
    doRestore = False
    if args.resume:
        snapshotToRestore = snapshot.copyFrom(args.resume)
        snapshotToRestore.verify()
        doRestore = True
        print 'We will restore the training from', args.resume
    
    # Command-line out dir takes priority, if not provided, we use the one in
    # the config file, if still not provided, we use the current directory
    outDir = args.out_dir
    if outDir is None and tlMsg.HasField(F_OUT_DIR):
        outDir = os.path.join(configDir, tlMsg.out_dir)
    elif outDir is None:
        outDir = '.'
    setattr(snapshot, 'out_filename',
            os.path.join(os.path.abspath(outDir), 
                         tlMsg.name + S_SNAP + E_PROTOTXT))
    print 'Will save all the results into', outDir
    
    # Command-line init weights take priority, if not provided, we use the ones
    # in the config file, if still not provided, we train from scratch
    prevModel = args.weights
    if prevModel is None and tlMsg.HasField(F_INIT_WEIGHTS):
        prevModel = os.path.join(configDir, tlMsg.init_weights)
    print 'Will initialize the training with the weights :', prevModel
    print 
    
    
    # For each multiSource message in the config file
    for idx, msMsg in enumerate(tlMsg.multi_source):
        
        # Check if this multisource message is ok
        assert msMsg.iterations > 0, \
        ' '.join(["The number of iterations should be > 0. (current:", \
                  str(msMsg.iterations), ")"])
        assert len(msMsg.stage) > 0, \
        ' '.join(["The config file has to provide at least one stage"])
        
        # If we resume this multiSource message
        if doRestore and snapshotToRestore.multisource > idx:
            print 'Restored the multisource', idx
            continue
        
        # Update the current snapshot
        setattr(snapshot, 'multisource', idx)
        print 'Will start the multisource', idx
        
        # Get the stages from the current multisource message
        initStage, stages = getStagesFromMsgs(msMsg, configDir, outDir)

        # Paths to the best models saved and their scores
        bestModels = [''] * (1 if args.clean else len(stages))
        bestScores = [0.] * (1 if args.clean else len(stages))
        
        # If we already computed some models and scores, restore them
        if doRestore and not snapshotToRestore.best_weights[0] is '':
            for i in range(len(snapshotToRestore.best_weights)):
                bestModels[i] = snapshotToRestore.best_weights[i]
                bestScores[i] = snapshotToRestore.best_scores[i]
        
        # For each iteration of this multiSource message
        for it in range(msMsg.iterations):
            # Check if we have to resume this iteration
            if doRestore and snapshotToRestore.iteration > it:
                print '> Restored the iteration', it
                continue
            
            # Update the current snapshot
            setattr(snapshot, 'iteration', it)
            setattr(snapshot, 'best_weights', [bestModels[i] for i
                                                    in range(len(bestModels))])
            setattr(snapshot, 'best_scores', [bestScores[i] for i 
                                                    in range(len(bestScores))])
            print '> Will start the iteration', it
            
            # Will store the models and scores generated by the list of stages
            # for this iteration
            nextResults = None
            
            # Check if we have to execute the init_stage
            allStages = stages
            if initStage is not None and it is 0:
                # On the first iteration, run initStage instead of first stage
                print '> Replace first stage by init stage'
                allStages = [initStage] + stages[1:]
            
            # Execute all the stages
            nextResults = executeListOfStages(allStages, prevModel, snapshot,
                                              snapshotToRestore, args.clean,
                                              args.quiet)
            
            # We restored the snapshot, if given
            if snapshotToRestore:
                doRestore = not snapshotToRestore.restored
            
            # Get the results
            nextModels, nextScores = zip(*nextResults)
            
            # Checking the results
            assert (len(nextModels) == len(nextScores)) and \
                   ((args.clean and len(nextModels) == 1) or \
                    (not args.clean and len(nextModels) == len(stages))), \
            ' '.join(["nextModels and nextScores' shape mismatch.\n", \
                      "nb of elements in nextModels :", \
                            str(len(nextModels)), "\n", \
                      "nb of elements in nextScores :", \
                            str(len(nextScores)), "\n", \
                      "expected nb of elements :", \
                            str("1" if args.clean else len(stages))])
            
            # Check if those results are better than previous ones
            for m in range(len(bestModels)):
                if not nextModels[m] and not nextScores[m]:
                    # No information about these weights, just pass
                    continue
                
                currNextScore = getScore(nextScores[m], msMsg.score_metric)
                
                # No scores given, so just assume it's better
                if currNextScore == bestScores[m] and currNextScore is 0:
                    bestModels[m] = nextModels[m]
                # If new score is better, change the best model
                elif currNextScore > bestScores[m]:
                    bestModels[m] = nextModels[m]
                    bestScores[m] = currNextScore
            
            # Input model of the next iteration is the best last-stage model
            prevModel = nextModels[-1]
            
            # Delete any models that are no longer needed 
            for model in nextModels:
                notNeeded = model not in bestModels and model != prevModel
                if model is not None and notNeeded and args.clean:
                    os.remove(model)
    
    print
    print 'Final models stored in', ', '.join(bestModels)
    raise SystemExit
    
