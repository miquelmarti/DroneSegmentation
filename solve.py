#!/usr/bin/env python


import caffe
from datetime import datetime
import score
import protoUtils

# Add the path to the layers to sys.path so caffe's code can import them
import fcnLayers
import inspect
import sys
import os
sys.path.append(os.path.dirname(inspect.getfile(fcnLayers)))

# Extensions
E_SOLVERSTATE = '.solverstate'


def runValidation(solver, numIter, layerNames):
    """Run the test step."""
    totalHist = None
    totalLoss = 0
    
    # Share the trained weights with the test net
    # TODO: What if multiple test_nets ? Here, we only use test_nets[0], 
    # TODO: numIter[0], ... a simple for should be ok, have to check
    solver.test_nets[0].share_with(solver.net)
    currIter = numIter[0]
    
    for i in range(currIter):
        # Run the network on its own data and labels
        # TODO: Implement image + gtImage
        _, hist, loss = score.runNetForward(solver.test_nets[0], 
                                            dataLayer=layerNames[0],
                                            lossLayer=layerNames[1],
                                            outLayer=layerNames[2],
                                            labelLayer=layerNames[3])
        if totalHist is None:
            totalHist = hist
        else:
            totalHist += hist
        totalLoss += loss
    
    scores = score.computeSegmentationScores(totalHist)
    scores.loss = totalLoss / currIter
    return scores


def printScores(scores, iteration):
    """Prints the given scores."""
    prefix = ' '.join(['>>>', str(datetime.now()), 'Iteration', 
                       str(iteration)])
    print prefix, 'loss', scores.loss
    print prefix, 'overall accuracy', scores.overallAcc
    print prefix, 'mean accuracy', scores.meanAcc
    print prefix, 'mean IU', scores.meanIu
    print prefix, 'fwavacc', scores.fwavacc


def validateAndPrint(solverParam, solver, layerNames, silent):
    """Runs a test and print the results."""
    scores = runValidation(solver, solverParam.testIter, layerNames)
    if not silent:
        printScores(scores, solver.iter)
    return scores

def saveSnapshotWeights(solverParam, solver, snapshot, silent):
    """Save a snapshot (for future resuming)."""
    solver.snapshot()
    
    # Update and save the snapshot
    if snapshot:
        snapshotFile = solverParam.snapshotPrefix + str(solver.iter) + \
                       E_SOLVERSTATE
        assert os.path.isfile(snapshotFile), \
        ' '.join(["Problem while creating the snapshot, nothing in", snapshotFile])
        setattr(snapshot, 'stage_snapshot', snapshotFile)
        snapshot.save(silent)

def doSnapOrTest(action, solverParam, solver, snapshot=None, layerNames=None, 
                 silent=False):
    """Execute the appropriate action."""
    if not action:
        return None
    elif action == 'SNAP':
        return saveSnapshotWeights(solverParam, solver, snapshot, silent)
    elif action == 'TEST':
        return validateAndPrint(solverParam, solver, layerNames, silent)
    else:
        print 'Unknown action, do nothing.. (', action, ')'
    
    return None

def checkHalt(prev, new, halt):
    """Check if we reach the halting criteria."""
    perc = (prev / new) if new != 0 else 0
    
    if (1. - perc) * 100 < halt:
        return True
    return False

def solveWithIntervals(solverParam, solver, halt, snapshot, layerNames, 
                       silent):
    """Run the needed number of iteration and do appropriate actions."""
    maxIter = solverParam.maxIter
    testInterval = solverParam.testInterval
    snapInterval = solverParam.snapshotInterval
    
    # If we don't have to snap or test
    snapInterval = (maxIter + 1) if snapInterval is 0 else snapInterval
    testInterval = (maxIter + 1) if testInterval is 0 else testInterval
    
    # Nb of iterations to do until next interval for snap and test
    rests = [snapInterval, testInterval]
    # Final scores to return
    scores = None
    # Tmp variable for computing the halt criteria
    prevScore = 0
    
    # Exit loop flag
    end = False
    
    while not end:
        # Get the number of iteration to do
        nbStepsToDo = 0
        if (rests[0] < rests[1]):
            nbStepsToDo = rests[0]
            rests = [snapInterval, rests[1] - nbStepsToDo]
        elif (rests[0] > rests[1]):
            nbStepsToDo = rests[1]
            rests = [rests[0] - nbStepsToDo, testInterval]
        else:
            nbStepsToDo = snapInterval
            rests = [snapInterval, testInterval]
        # If the maxIter has been reached (and no halt criteria)
        if halt is None and (solver.iter + nbStepsToDo) >= maxIter:
            nbStepsToDo = maxIter - solver.iter
                
        # Do them
        solver.step(nbStepsToDo)
        
        # Execute the appropriate actions
        if solver.iter % snapInterval is 0:
            doSnapOrTest('SNAP', solverParam, solver, snapshot=snapshot)
        if solver.iter % testInterval is 0:
            scores = doSnapOrTest('TEST', solverParam, solver,
                                  layerNames=layerNames, silent=silent)
        
        # Check if we have to quit the loop
        if not halt is None and scores:
            end = checkHalt(prevScore, scores.meanIu, halt)
            prevScore = scores.meanIu
        elif solver.iter == maxIter:
            end = True
    
    # If we have to do a last test
    if testInterval <= maxIter and rests[1] != testInterval:
        scores = doSnapOrTest('TEST', solverParam, solver, 
                              layerNames=layerNames, silent=silent)
    
    # If we used the halt criteria
    if not halt is None and not silent:
        print ' '.join(['Halting ratio', str(halt), 'acheived in', 
                        str(solver.iter), 'iterations.'])
    
    return scores


def solve(solverParam, modelFilename=None, preProcFun=None, haltPercent=None,
          layerNames=('data', 'loss', 'out', 'label'), snapshot=None,
          snapshotToRestore=None, silent=False):
    """Solve a solver with given parameters."""
    
    # Use the absolute path, since we change the directory later
    # TODO: We do ?
    if modelFilename is not None:
        modelFilename = os.path.abspath(modelFilename)
    
    # Pycaffe requires we be in the directory where solver.prototxt is
    startDir = os.getcwd()
    os.chdir(solverParam.directory)
    
    # Get solver
    solver = caffe.get_solver(str(os.path.basename(solverParam.filename)))
    
    # Load the weights or restore the provided solverstate
    if snapshotToRestore and not snapshotToRestore.restored \
                         and snapshotToRestore.stage_snapshot:
        solver.restore(str(snapshotToRestore.stage_snapshot))
    elif modelFilename is not None:
        solver.net.copy_from(str(modelFilename))
    
    # Pre-processing function
    # TODO: not flexible, works only for fcnSurgery
    if preProcFun is not None:
        preProcFun(solver.net)
    
    # Will store the new scores for this stage
    latestScores = None
    
    # Execute all the iterations, taking care of the intervals
    # TODO: Make it flexible to dynamic number of intervals
    latestScores = solveWithIntervals(solverParam, solver, haltPercent,
                                      snapshot, layerNames, silent)
            
    # Clean up environment and return the final testing results.
    os.chdir(startDir)
    return solver.net, latestScores


