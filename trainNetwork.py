#!/usr/bin/env python

#  imported so that we can train with these python layers.
from fcnLayers import *
import argparse
import os
import caffe
from caffeUtils import solve, fcnSurgery

SOLVER_FILENAME = 'solver.prototxt'


def getArguments():
    '''
    If running solve.py from command line, read in command-line arguments.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='\
    Weights to use to initialize the network.')
    parser.add_argument('--device', type=int, default=0, help='\
    ID of the GPU device to train.')
    parser.add_argument('--noval', action='store_true', help="\
    Don't score on the Pascal test set.")
    parser.add_argument('--fcn', action='store_true', help="\
    Apply FCN-style net surgery to the network before solving.")
    parser.add_argument('--halt_percent', type=float, help="\
    If the metric score on the test_net differs by less than this percentage \
    value between tests, halt.  If not provided, continue for iterations \
    specified by solver file's max_iter.")
    parser.add_argument('learn_dir', type=str,
                        help='Directory containing training .prototxt files')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = getArguments()
    preProcFun = None
    if args.fcn:
        preProcFun = fcnSurgery.fcnInterp
        
    if args.device is not None:
        caffe.set_device(args.device)
        caffe.set_mode_gpu()
        
    os.chdir(args.learn_dir)
    solve.solve(SOLVER_FILENAME, args.weights, preProcFun=preProcFun,
                haltPercent=args.halt_percent)
