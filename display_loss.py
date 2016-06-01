#!/usr/bin/env python

# Takes, as inputs, the file with all the losses and the frequency of the display (display parameter in the solver.prototxt used for the training). Plot the loss then, with matplotlib.
# Use : python display_loss.py loss.txt 20


import argparse
import string
import os
import matplotlib.pyplot as plt
import numpy as np

ITER_TAG = "Iteration"
TRN_LOSS_TAG = "loss = "
TST_LOSS_TAG = "loss"
OV_ACC_TAG = "overall accuracy"
MEAN_ACC_TAG = "mean accuracy"
MEAN_IU_TAG = "mean IU"
FWAVACC_TAG = "fwavacc"
TRAIN_LOSS_TAGS = [ITER_TAG, TRN_LOSS_TAG]

SEG_TEST_PREFIX = '>>>'
SEG_TEST_AVOID = '.cpp'
MS_ITER_START_STR = 'starting multi-source iteration'
STAGE_START_STR = '-> Execute stage'
OUT_MODEL_STR = '-> Produce the model'
BEST_SCORE_STR = 'New best score for stage'


def isTrainLossLine(line):
    return (not line.startswith(SEG_TEST_PREFIX) and
            SEG_TEST_AVOID in line and
            all([tag in line for tag in TRAIN_LOSS_TAGS]))


def getIterNum(line):
    '''Returns the iteration number that this line corresponds to.

    Throws an exception if the line contains no iteration information.
    '''
    return getTaggedValue(line, ITER_TAG, valType=int)


def getTaggedValue(line, tag, valType=float):
    postTagStr = line.partition(tag)[2]
    valStr = postTagStr.strip().split(' ')[0]
    cleanValStr = valStr.strip(string.punctuation)
    return valType(cleanValStr)


def getTestStats(lines):
    tags = [TST_LOSS_TAG,
            OV_ACC_TAG,
            MEAN_ACC_TAG,
            MEAN_IU_TAG,
            FWAVACC_TAG]
    stats = {tag: [] for tag in tags}
    # accumulate statistic values in the tags dict
    for l in lines:
        tagsInLine = filter(lambda tag: tag in l, tags)
        if len(tagsInLine) == 0:
            continue
        tagVals = {tag: getTaggedValue(l, tag) for tag in tagsInLine}
        iteration = getIterNum(l)
        for tag in tagVals:
            stats[tag].append((iteration, tagVals[tag]))
    for tag in stats:
        stats[tag] = sorted(stats[tag], key=lambda x: x[0])
    return stats


def plotXAndY(x, y, title):
    plt.plot(x, y)
    plt.title(title)
    plt.show()


def plotMultiple(xs, ys, labels, title):
    for x, y, label in zip(xs, ys, labels):
        plt.plot(x, y, label=label)
    plt.title(title)
    plt.legend(bbox_to_anchor=(0.7, 1), loc=2, borderaxespad=0.)
    plt.show()
    

def plotLearningLogs(lossLines, segTestLines, prefix=''):
    # sort the lines by their date stamps
    plotTrainingLosses(lossLines, prefix)
    plotSegTests(segTestLines, prefix)


def plotTrainingLosses(lossLines, prefix=''):
    iterNums = map(getIterNum, lossLines)
    losses = [float(l.strip().rpartition(' ')[2]) for l in lossLines]
    plotXAndY(iterNums, losses, prefix + 'training loss')


def plotSegTests(segTestLines, prefix=''):
    if len(segTestLines) is 0:
        return
    testStats = getTestStats(segTestLines)
    # plot loss separately, since it exists on a very different scale
    testLossIters, testLosses = zip(*testStats[TST_LOSS_TAG])
    plotXAndY(testLossIters, testLosses, prefix + 'test loss')
    del(testStats[TST_LOSS_TAG])

    # plot other statistics
    testIters, testVals = [], []
    for tag in testStats:
        statIters, statVals = zip(*testStats[tag])
        testIters.append(statIters)
        testVals.append(statVals)
    plotMultiple(testIters, testVals, testStats.keys(), prefix + 'stats')


def plotSmoothedLossCurves(logList, avgOverLast=10, interval=1,
                           title='Loss curves'):
    allSmoothIters, allSmoothLosses, allNames = [], [], []
    for logFile in logList:
        with open(logFile, 'r') as f:
            # extract the loss values and their iterations
            points = [(getIterNum(l), getTaggedValue(l, TRN_LOSS_TAG))
                      for l in f if isTrainLossLine(l)]
            iters, losses = zip(*points)
            
            # Compute the average loss over the last avgOverLast losses
            allSmoothIters.append(iters[avgOverLast-1::interval])
            sampleIdxs = range(avgOverLast, len(points) + 1, interval)
            smoothLoss = [np.mean(losses[i-avgOverLast:i]) for i in sampleIdxs]
            allSmoothLosses.append(smoothLoss)
            allNames.append(os.path.basename(logFile))
            
    # produce smoothedPlotSequences from lossSequences
    plotMultiple(allSmoothIters, allSmoothLosses, allNames, title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="\
    Plots statistics produced by the fcn.berkeleyvision.org solving process.  \
    Arguments may be provided to select which statistics to display.  If no \
    arguments are provided, all statistics will be shown.")
    # parser.add_argument('-1', '--merge', action="store_true",
    #                     help='Plot stats from all multi-source stages on a \
    #                     single graph.')
    parser.add_argument('log_file', type=str,
                        help='Path to the Caffe output log file')
    args = parser.parse_args()

    with open(args.log_file, 'r') as f:
        lossLines = []
        segTestLines = []
        bestScoreLines = []
        msIter = None
        stage = ''
        prefix = ''
        currentIter = 0

        for l in f:
            if l.startswith(OUT_MODEL_STR):
                # seg-test messages for the job are done, so plot them
                plotSegTests(segTestLines, prefix)
                segTestLines = []

            # these cases are simply for labelling of plots
            elif l.startswith(MS_ITER_START_STR):
                postStr = l.partition(MS_ITER_START_STR)[2]
                msIter = int(postStr.split()[0])
            elif l.startswith(STAGE_START_STR):
                stage = l.partition(STAGE_START_STR)[2].strip()
                prefix = 'MS Iter. ' + str(msIter) + ', ' + stage + ': '
            # remaining cases are aggregating data
            elif l.startswith(BEST_SCORE_STR):
                bestScoreLines.append(l)
            elif not (l.startswith(SEG_TEST_PREFIX) or SEG_TEST_AVOID in l):
                continue  # ignore badly-formatted lines
            elif l.startswith(SEG_TEST_PREFIX) and SEG_TEST_AVOID not in l:
                segTestLines.append(l)
            elif all([t in l for t in TRAIN_LOSS_TAGS]):
                # if iteration goes down, we must be in a new job
                newIter = getIterNum(l)
                if newIter < currentIter:
                    plotTrainingLosses(lossLines, prefix)
                    lossLines = []
                lossLines.append(l)
                currentIter = newIter

        # display the remaining collected values
        if msIter is None:
            prefix = ''
        plotTrainingLosses(lossLines, prefix)
        plotSegTests(segTestLines, prefix)

        if len(bestScoreLines) > 0:
            bestScores = [float(l.partition(':')[2]) for l in bestScoreLines]
            firstStage = bestScoreLines[0].partition(':')[0].split()[-1]
            plt.plot(range(len(bestScores)), bestScores, 'bs')
            plt.title('starting at ' + firstStage)
            plt.show()
