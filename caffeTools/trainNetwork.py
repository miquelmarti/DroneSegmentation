#!/usr/bin/env python

#  imported so that we can train with these python layers.
import argparse
import caffe
from caffeUtils import solve, fcnSurgery, solverParam


def getArguments():
    '''
    If running solve.py from command line, read in command-line arguments.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', type=str, help="\
    Weights to use to initialize the network.")
    parser.add_argument('-d', '--device', type=int, default=0, help="\
    ID of the GPU device to train.")
    parser.add_argument('-f', '--fcn', action='store_true', help="\
    Apply FCN-style net surgery to the network before solving.")
    parser.add_argument('-p', '--halt_percent', type=float, help="\
    If the metric score on the test_net differs by less than this percentage \
    value between tests, halt.  If not provided, continue for iterations \
    specified by solver file's max_iter.")
    parser.add_argument('-o', '--out_file', help='\
    A file in which to save the final caffe model.')
    parser.add_argument('solver', help="\
    The name of the solver prototxt file.  If omitted, we assume the name is \
    solver.prototxt.")
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
    
    solverParam = solverParam.SolverParam(args.solver)
    solverParam.verify()
        
    net, scores = solve.solve(solverParam, args.weights, preProcFun,
                              args.halt_percent, ('data', 'loss', 'score', 'label'))
    if args.out_file is not None:
        net.save(args.out_file)
