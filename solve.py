#!/usr/bin/env python

import caffe
from datetime import datetime
import score
import protoUtils


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
    
    solver = caffe.get_solver(solverFilename)
    if modelFilename is not None:
        solver.net.copy_from(modelFilename)

    # surgeries
    if preProcFun is not None:
        preProcFun(solver.net)

    solverSpec = protoUtils.readSolver(solverFilename)
    maxIter = solverSpec.max_iter
    testInterval = solverSpec.test_interval
    testIter = solverSpec.test_iter[0]
    if not testInterval > 0:
        raise ValueError("test_interval is invalid (" + str(testInterval) +
                         ").  Is it specified in " + solverFilename + "?")

    latestScores = None
    if haltPercent is None:
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
        prevScore = 0  # assuming this is mean IU for now.
        newScore = 0
        while True:
            solver.step(testInterval)
            scores = validateAndPrint(solver, testIter, silent,
                                      outLayer='score',
                                      lossLayer='loss',
                                      labelLayer='label')
            newScore = scores.meanIu
            percentIncrease = 1. - (prevScore/newScore)
            if percentIncrease < haltPercent:
                break
            prevScore = newScore

        if not silent:
            outStr = ' '.join(['Halting ratio', str(haltPercent),
                               'acheived in', str(solver.iter), 'iterations.'])
            print outStr
            
    if outModelFilename:
        solver.net.save(str(outModelFilename))

    # return the final testing results.
    return latestScores
