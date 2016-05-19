#!/usr/bin/env python

import caffe
import argparse
import datetime

import score
import protoUtils


def getArguments():
    '''
    If running solve.py from command line, read in command-line arguments.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelFilename', type=str, help='\
    Weights to use to initialize the network.')
    parser.add_argument('--device', type=int, default=0, help='\
    ID of the GPU device to train.')
    parser.add_argument('--noval', action='store_true', help="\
    Don't score on the Pascal test set.")
    parser.add_argument('--fcn', action='store_true', help="\
    Apply FCN-style net surgery to the network before solving.")
    parser.add_argument('--halt_ratio', type=float, help="\
    If the metric score on the test_net differs by less than this value \
    between tests, halt.  If not provided, continue for iterations specified \
    in max_iter.")

    parser.add_argument('solver',
                        help='Filename of the solver prototxt file.')
    args = parser.parse_args()
    return args


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
          haltPercent=None, outModelFilename=None):
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
            printScores(runValidation(solver, testIter), solver.iter)
    else:
        prevScore = 0  # assuming this is mean IU for now.
        newScore = 0
        while True:
            solver.step(testInterval)
            scores = runValidation(solver, testIter)
            printScores(scores, solver.iter)
            newScore = scores.meanIu
            percentIncrease = 1. - (prevScore/newScore)
            if percentIncrease < haltPercent:
                break
            prevScore = newScore

        outStr = ' '.join(['Halting ratio', str(haltPercent),
                           'acheived after', str(solver.iter), 'iterations.'])
        print outStr


if __name__ == "__main__":
    args = getArguments()
    preProcFun = None
    if args.fcn:
        import fcnSurgery
    solve(args.solver, args.modelFilename, preProcFun, args.device,
          args.haltPercent)
