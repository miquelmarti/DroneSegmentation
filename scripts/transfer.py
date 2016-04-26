#!/usr/bin/env python

"""Carries out transfer learning according to a provided configuration file."""

# auto-generated python classes for protobuf formats
import transferLearning_pb2
import caffe
import argparse
import os
import caffeUtils
from stage import PrototxtStage, CommandStage


FILENAME_FIELD = 'filename'
NET_FILENAME_FIELD = 'net_filename'
SOLVE_CMD_FIELD = 'command'


def getArguments():
    """Defines and parses command-line arguments to this script."""
    parser = argparse.ArgumentParser()
    # optional arguments
    parser.add_argument('--clean', action="store_true", help='\
    Cleans up intermediate .prototxt and .caffemodel files as the script \
    finishes with them.')
    parser.add_argument('--model', help='\
    A .caffemodel file containing the initial weights of the first stage.  \
    If not provided, the first stage will learn all weights from scratch.')
    # TODO argument to allow user to specify version of caffe to use
    machineGroup = parser.add_mutually_exclusive_group()
    machineGroup.add_argument('--cpu', action='store_true', help='\
    If this flag is set, runs all training on the CPU.')
    machineGroup.add_argument('--gpu', type=int, default=0, help='\
    Allows the user to specify which GPU training will run on.')

    # required arguments
    parser.add_argument('stages', help='\
    A .prototxt file defining the transfer learning stages to be performed.')
    return parser.parse_args()


def getStagesFromMsgs(stageMsgs, solverFilename=None, trainNetFilename=None):
    """Instantiates a sequence of stages from protobuf "stage" messages."""
    stages = []
    for stageMsg in stageMsgs:
        # unpack values
        newStage = None
        if stageMsg.type == transferLearning_pb2.Stage.PROTOTXT:
            if stageMsg.prototxt_solver.HasField(FILENAME_FIELD):
                solverFilename = stageMsg.prototxt_solver.filename
            elif solverFilename is None:
                raise Exception('First training stage provides no solver file')
            newStage = PrototxtStage(stageMsg.name, solverFilename,
                                     stageMsg.freeze, stageMsg.ignore,
                                     trainNetFilename)
        elif stageMsg.type == transferLearning_pb2.Stage.COMMAND:
            trainNetFilename = None
            if stageMsg.cmd_solver.HasField(NET_FILENAME_FIELD):
                trainNetFilename = stageMsg.cmd_solver.net_filename
            newStage = CommandStage(stageMsg.name, stageMsg.cmd_solver.command,
                                    stageMsg.cmd_solver.out_model_filename,
                                    trainNetFilename, stageMsg.freeze,
                                    stageMsg.ignore)
        stages.append(newStage)
    return stages


def executeListOfStages(stages, firstModel, clean):
    model = firstModel
    for stage in stages:
        newModel = stage.execute(model)
        if clean and model != firstModel:
            os.remove(model)
        model = newModel
    return model


if __name__ == "__main__":
    args = getArguments()
    if args.cpu:
        caffe.set_mode_cpu()
    else:
        caffe.set_device(args.gpu)
        caffe.set_mode_gpu()

    # Read in the stages and carry them out
    tlMsg = transferLearning_pb2.TransferLearning()
    caffeUtils.readFromPrototxt(tlMsg, args.stages)
    stageMsgs = tlMsg.stage
    stages = getStagesFromMsgs(stageMsgs)
    executeListOfStages(stages, args.model, args.clean)
