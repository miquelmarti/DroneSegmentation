#!/usr/bin/env python

# Takes, as inputs, the file with all the losses and the frequency of the display (display parameter in the solver.prototxt used for the training). Plot the loss then, with matplotlib.
# Use : python display_loss.py loss.txt 20


import argparse
import numpy as np
import matplotlib.pyplot as plt
import string

ITER_TAG = "Iteration"
LOSS_TAG = "loss"
OV_ACC_TAG = "overall accuracy"
MEAN_ACC_TAG = "mean accuracy"
MEAN_IU_TAG = "mean IU"
FWAVACC_TAG = "fwavacc"

TRN_LOSS_ARG = "train_loss"
TST_LOSS_ARG = "val_loss"
OV_ACC_ARG = "ov_acc"
MEAN_ACC_ARG = "mean_acc"
MEAN_IU_ARG = "mean_iu"
FWAVACC_ARG = "fwavacc"

TAGS_TO_ARGS = {LOSS_TAG: TST_LOSS_ARG,
                OV_ACC_TAG: OV_ACC_ARG,
                MEAN_ACC_TAG: MEAN_ACC_ARG,
                MEAN_IU_TAG: MEAN_IU_ARG,
                FWAVACC_TAG: FWAVACC_ARG}
TRAIN_LOSS_TAGS = [ITER_TAG, LOSS_TAG]
SEG_TEST_PREFIX = '>>>'
SEG_TEST_AVOID = '.cpp'


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
    tags = [LOSS_TAG,
            OV_ACC_TAG,
            MEAN_ACC_TAG,
            MEAN_IU_TAG,
            FWAVACC_TAG]
    stats = { tag: [] for tag in tags }
    # accumulate statistic values in the tags dict
    for l in lines:
        tagsInLine = filter(lambda tag : tag in l, tags)
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
    
    
parser = argparse.ArgumentParser(description="\
Plots statistics produced by the fcn.berkeleyvision.org solving process.  \
Arguments may be provided to select which statistics to display.  If no \
arguments are provided, all statistics will be shown.")
parser.add_argument('log_file', type=str,
                    help='Path to the Caffe output log file')
disp_group = parser.add_argument_group('Graphs to display')
disp_group.add_argument('--' + TRN_LOSS_ARG, action="store_true",
                    help='Display the training loss.')
disp_group.add_argument('--' + TST_LOSS_ARG, action="store_true",
                    help='Display the validation loss.')
disp_group.add_argument('--' + OV_ACC_ARG, action="store_true",
                    help='Display the overall validation accuracy.')
disp_group.add_argument('--' + MEAN_ACC_ARG, action="store_true",
                    help='Display the mean validation accuracy.')
disp_group.add_argument('--' + MEAN_IU_ARG, action="store_true",
                    help='Display the validation mean IU.')
disp_group.add_argument('--' + FWAVACC_ARG, action="store_true",
                    help='Display the fwavacc (?) on the validation set.')
args = parser.parse_args()

argVars = vars(args)
displayArgVals = [argVars[arg] for tag, arg in TAGS_TO_ARGS.iteritems()]
displayAll = not reduce(lambda x,y: x or y, displayArgVals)

with open(args.log_file, 'r') as f:
    lossLines = []
    segTestLines = []
    for l in f:
        if not (l.startswith(SEG_TEST_PREFIX) or SEG_TEST_AVOID in l):
            continue # ignore badly-formatted lines
        elif l.startswith(SEG_TEST_PREFIX) and SEG_TEST_AVOID not in l:
            segTestLines.append(l)
        elif reduce(lambda x,y: x and y, [t in l for t in TRAIN_LOSS_TAGS]):
            lossLines.append(l)
    iterNums = map(getIterNum, lossLines)
    losses = [float(l.strip().rpartition(' ')[2]) for l in lossLines]
    if args.train_loss or displayAll:
        plotXAndY(iterNums, losses, 'training loss')

    testStats = getTestStats(segTestLines)
    for tag in testStats:
        if displayAll or argVars[TAGS_TO_ARGS[tag]]:
            iterations, stats = zip(*testStats[tag])
            plotXAndY(iterations, stats, tag)
