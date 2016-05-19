#!/usr/bin/env python

import caffe
import datetime
import score
import protoUtils


def runValidation(solver, numIter, lossLayer='loss', outLayer='score',
                  labelLayer='label'):
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
    

def solve(solverFilename, modelFilename, preProcFun=None, device=None,
          haltPercent=None, outModelFilename=None, silent=False):
    if device is not None:
        caffe.set_device(device)
        caffe.set_mode_gpu()
    
    solver = caffe.get_solver(solverFilename)
    if modelFilename is not None:
        solver.net.copy_from(modelFilename)

    # surgeries
    if preProcFun is not None:
        preProcFun(solver.net)

    solverSpec = protoUtils.readSolver(solverFilename)
    maxIter = solverSpec.max_iter
    testInterval = solverSpec.test_interval
    testIter = solverSpec.test_iter
    if not testInterval > 0:
        raise ValueError("test_interval is invalid (" + str(testInterval) +
                         ").  Is it specified in " + solverFilename + "?")
        
    if haltPercent is None:
        for _ in range(maxIter):
            solver.step(testInterval)
            if not silent:
                printScores(runValidation(solver, testIter), solver.iter)
    else:
        prevScore = 0  # assuming this is mean IU for now.
        newScore = 0
        while True:
            solver.step(testInterval)
            scores = runValidation(solver, testIter)
            if not silent:
                printScores(scores, solver.iter)
            newScore = scores.meanIu
            percentIncrease = 1. - (prevScore/newScore)
            if percentIncrease < haltPercent:
                break
            prevScore = newScore

        if not silent:
            outStr = ' '.join(['Halting ratio', str(haltPercent),
                               'acheived in', str(solver.iter), 'iterations.'])
            print outStr
