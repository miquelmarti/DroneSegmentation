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

TRAIN_NET_FIELD = 'train_net'
TEST_NET_FIELD = 'test_net'


def runValidation(solver, numIter, dataLayer, lossLayer, outLayer, labelLayer):
    totalHist = None
    totalLoss = 0
    
    # Share the trained weights with the test net
    solver.test_nets[0].share_with(solver.net)
    
    for i in range(numIter):
        # Run the network on its own data and labels
        # TODO: Implement image + gtImage
        _, hist, loss = score.runNetForward(solver.test_nets[0], 
                                            dataLayer=dataLayer,
                                            lossLayer=lossLayer,
                                            outLayer=outLayer,
                                            labelLayer=labelLayer)
        if totalHist is None:
            totalHist = hist
        else:
            totalHist += hist
        totalLoss += loss
    
    scores = score.computeSegmentationScores(totalHist)
    scores.loss = totalLoss / numIter
    return scores


def printScores(scores, iteration):
    prefix = ' '.join(['>>>', str(datetime.now()), 'Iteration',
                       str(iteration)])
    print prefix, 'loss', scores.loss
    print prefix, 'overall accuracy', scores.overallAcc
    print prefix, 'mean accuracy', scores.meanAcc
    print prefix, 'mean IU', scores.meanIu
    print prefix, 'fwavacc', scores.fwavacc


def validateAndPrint(solver, testIter, silent, dataLayer, lossLayer, outLayer,
                     labelLayer):
    scores = runValidation(solver, testIter, dataLayer, lossLayer, outLayer,
                           labelLayer)
    if not silent:
        printScores(scores, solver.iter)
    return scores


# TODO: implement with silent
def solve(solverFilename, modelFilename=None, preProcFun=None,
          haltPercent=None, dataLayer='data', lossLayer='loss', outLayer='out',
          labelLayer='label', silent=False):
    
    # Use the absolute path, since we change the directory later
    if modelFilename is not None:
        modelFilename = os.path.abspath(modelFilename)
    
    # Pycaffe requires we be in the directory where solver.prototxt is
    startDir = os.getcwd()
    os.chdir(os.path.dirname(solverFilename))
    solverFilename = os.path.basename(solverFilename)
    
    # Load the weights
    solver = caffe.get_solver(str(solverFilename))
    if modelFilename is not None:
        solver.net.copy_from(str(modelFilename))
    
    # Pre-processing function
    # TODO: not flexible, works only for surgery
    if preProcFun is not None:
        preProcFun(solver.net)
    
    # Check the solver
    # TODO: Do more assertions for the solver
    # TODO:     - testInterval < maxIter
    # TODO: By the way, perhaps it should be done before
    solverSpec = protoUtils.readSolver(solverFilename)
    maxIter = solverSpec.max_iter
    testInterval = solverSpec.test_interval
    assert not testInterval < 0, \
    ' '.join(["test_interval is invalid (", testInterval,
              ").  Is it specified in ", solverFilename, "?"])
    
    # Will store the new scores for this stage
    latestScores = None
    
    # Solve the network and do the validation step if needed
    # TODO: SOULD be ok, but have to test
    if testInterval is 0 or len(solver.test_nets) is 0:
        solver.solve()
    else:
        # At least one test net was specified, so run it at the given interval
        # TODO: Only deal with one of the test_net, the solver can handle more
        testIter = solverSpec.test_iter[0]
        
        # If the halt criteria is not provided
        # TODO: Compact here
        if haltPercent is None:
            # run for testInterval iterations, then test
            for _ in range(maxIter / testInterval):
                solver.step(testInterval)
                latestScores = validateAndPrint(solver, testIter, silent,
                                                dataLayer, lossLayer,
                                                outLayer, labelLayer)

            # Finish up any remaining steps
            if maxIter % testInterval != 0:
                solver.step(maxIter % testInterval)
                latestScores = validateAndPrint(solver, testIter, silent,
                                                dataLayer, lossLayer,
                                                outLayer, labelLayer)

        else:
            # Stop when the mean IU improvement drops below given percentage.
            prevScore = 0  # Assuming this is mean IU for now.
            newScore = 0
            while True:
                solver.step(testInterval)
                scores = validateAndPrint(solver, testIter, silent,
                                          dataLayer, lossLayer,
                                          outLayer, labelLayer)
                newScore = scores.meanIu
                perc = 0
                if not newScore is 0:
                    perc = prevScore/newScore
                percentIncrease = (1. - perc) * 100
                if percentIncrease < haltPercent:
                    print percentIncrease, ' is less than ', haltPercent
                    break
                prevScore = newScore

            if not silent:
                print ' '.join(['Halting ratio', str(haltPercent),
                                'acheived in', str(solver.iter),
                                'iterations.'])
            
    # Clean up environment and return the final testing results.
    os.chdir(startDir)
    return solver.net, latestScores


