#!/usr/bin/env python

import argparse
import solve


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
    parser.add_argument('--halt_percent', type=float, help="\
    If the metric score on the test_net differs by less than this percentage \
    value between tests, halt.  If not provided, continue for iterations \
    specified by solver file's max_iter.")

    parser.add_argument('solver',
                        help='Filename of the solver prototxt file.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = getArguments()
    preProcFun = None
    if args.fcn:
        import fcnSurgery
        preProcFun = fcnSurgery.fcnInterp
    solve.solve(args.solver, args.modelFilename, preProcFun, args.device,
                args.halt_percent, silent=False)
