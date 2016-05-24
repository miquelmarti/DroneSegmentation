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


# TODO add data layer parameter
def runValidation(solver, numIter, outLayer, lossLayer, labelLayer):
    totalHist = None
    totalLoss = 0
    # share the trained weights with the test net
    solver.test_nets[0].share_with(solver.net)
    for i in range(numIter):
        # run the network on its own data and labels
        _, hist, loss = score.runNetForward(solver.test_nets[0],
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


# TODO add data layer parameter
def validateAndPrint(solver, testIter, silent, outLayer, lossLayer,
                     labelLayer):
    scores = runValidation(solver, testIter, outLayer, lossLayer, labelLayer)
    if not silent:
        printScores(scores, solver.iter)
    return scores

    
# TODO add data layer parameter
def solve(solverFilename, modelFilename, outModelFilename=None,
          preProcFun=None, haltPercent=None, silent=False,
          outLayer='score', lossLayer='loss', labelLayer='label'):
    # pycaffe requires we be in the directory where solver.prototxt lives
    startDir = os.getcwd()
    os.chdir(os.path.dirname(solverFilename))
    solverFilename = os.path.basename(solverFilename)
    
    solver = caffe.get_solver(str(solverFilename))
    if modelFilename is not None:
        solver.net.copy_from(str(modelFilename))

    # surgeries
    if preProcFun is not None:
        preProcFun(solver.net)

    solverSpec = protoUtils.readSolver(solverFilename)
    maxIter = solverSpec.max_iter
    # we assume here that the user specifies separate test and train nets.
    testInterval = solverSpec.test_interval
    if not testInterval > 0:
        raise ValueError("test_interval is invalid (" + str(testInterval) +
                         ").  Is it specified in " + solverFilename + "?")

    latestScores = None
    if len(solverSpec.test_net) is 0:
        # no test nets specified - just run the solver normally.
        solver.solve()

    # test nets were specified, so behave accordingly
    else:
        testIter = solverSpec.test_iter[0]
        if haltPercent is None:
            # run for testInterval iterations, then test
            for _ in range(maxIter / testInterval):
                solver.step(testInterval)
                latestScores = validateAndPrint(solver, testIter, silent,
                                                outLayer='score',
                                                lossLayer='loss',
                                                labelLayer='label')

            # Finish up any remaining steps
            if maxIter % testInterval != 0:
                solver.step(maxIter % testInterval)
                latestScores = validateAndPrint(solver, testIter, silent,
                                                outLayer='score',
                                                lossLayer='loss',
                                                labelLayer='label')

        else:
            # stop when the mean IU improvement drops below given percentage.
            prevScore = 0  # assuming this is mean IU for now.
            newScore = 0
            while True:
                solver.step(testInterval)
                scores = validateAndPrint(solver, testIter, silent,
                                          outLayer='score',
                                          lossLayer='loss',
                                          labelLayer='label')
                newScore = scores.meanIu
                percentIncrease = (1. - (prevScore/newScore)) * 100
                if percentIncrease < haltPercent:
                    print percentIncrease, ' is less than ', haltPercent
                    break
                prevScore = newScore

            if not silent:
                print ' '.join(['Halting ratio', str(haltPercent),
                                'acheived in', str(solver.iter),
                                'iterations.'])
            
    if outModelFilename:
        solver.net.save(str(outModelFilename))

    # Clean up environment and return the final testing results.
    os.chdir(startDir)
    return latestScores
