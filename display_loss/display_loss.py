#!/usr/bin/env python

# Takes, as inputs, the file with all the losses and the frequency of the display (display parameter in the solver.prototxt used for the training). Plot the loss then, with matplotlib.
# Use : python display_loss.py loss.txt 20


import argparse
import numpy as np
import matplotlib.pyplot as plt
import operator


TARGET_STRS = ['Iteration', 'loss']
ITER_VAL_IDX = 5
LOSS_VAL_IDX = 9


parser = argparse.ArgumentParser()
# Mandatory options
parser.add_argument('log_file', type=str, help='Path to the Caffe output log file')
# parser.add_argument('display_frequency', type=int, help='Number between each displayed iteration')

args = parser.parse_args()

with open(args.log_file, 'r') as f:
    
    lossLines = [l for l in f if reduce(lambda x,y: x and y,
                                        [t in l for t in TARGET_STRS])]
    getIterFn = lambda l: int(l.split(' ')[ITER_VAL_IDX].strip(','))
    iterNums = [getIterFn(l) for l in lossLines]
    frequency = getIterFn(lossLines[1]) - getIterFn(lossLines[0])
    losses = [float(l.strip().rpartition(' ')[2]) for l in lossLines]
    x = np.arange(0, len(losses)*frequency, frequency)
    plt.plot(x, losses)
    plt.show()
