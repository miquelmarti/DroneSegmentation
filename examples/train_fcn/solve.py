#!/usr/bin/env python

import caffe

import numpy as np
import os
import sys
import argparse

from caffeUtils import surgery


def getArguments():
    """Defines and parses command-line arguments to this script."""
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument('--solver', type=str,
                        help='Directory containing training .prototxt files')
    parser.add_argument('--weights', type=str, default=None,
                        help='Weights to use to initialize the network')

    # optional arguments
    parser.add_argument('--mode', type=int, default=0,
                        help='CPU or GPU mode, -1 for CPU, nb of device for GPU')
    parser.add_argument('--nb_iter', type=int, default=1000,
                        help='Number of iterations')
    
    return parser.parse_args()


if __name__ == "__main__":
    
    # Get the arguments 
    args = getArguments()
    
    # Set CPU or GPU mode
    if args.mode == -1:
        caffe.set_mode_cpu()
    else:
        caffe.set_device(args.mode)
        caffe.set_mode_gpu()

    # Get solver and copy weights if given
    solver = caffe.SGDSolver(os.path.abspath(args.solver))
    if args.weights is not None:
        solver.net.copy_from(os.path.abspath(args.weights))
    
    # Surgeries
    interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
    surgery.interp(solver.net, interp_layers)

    # scoring
    #val = np.loadtxt('/home/shared/datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/seg11valid.txt', dtype=str)

    solver.step(args.nb_iter)

    #for _ in range(25):
    #    solver.step(4000)
    #    score.seg_tests(solver, False, val, layer='score')


