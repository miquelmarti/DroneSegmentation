#!/usr/bin/env python

import caffe
import surgery
import score

import numpy as np
import os
import sys
import argparse
import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))

# weights = '../segnet/VGG_ILSVRC_16_layers.caffemodel'

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str,
                    help='Weights to use to initialize the network')
parser.add_argument('--learn_dir', type=str,
                    help='Directory containing training .prototxt files')
parser.add_argument('--device', type=int, default=0,
                    help='ID of the GPU device to train.')
args = parser.parse_args()

# init
caffe.set_device(args.device)
caffe.set_mode_gpu()

absWeightsPath = os.path.abspath(args.weights)
if args.learn_dir is not None:
    os.chdir(args.learn_dir)
solver = caffe.SGDSolver(os.path.join('solver.prototxt'))
if args.weights is not None:
    solver.net.copy_from(absWeightsPath)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('/home/shared/datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/seg11valid.txt', dtype=str)

for _ in range(25):
    solver.step(4000)
    score.seg_tests(solver, False, val, layer='score')
