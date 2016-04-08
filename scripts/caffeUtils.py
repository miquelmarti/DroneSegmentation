"""A collection of utility functions for protocol buffers and caffe."""

import google.protobuf
from caffe.proto import caffe_pb2

MODEL_FIELD = 'net'

def getModelFilename(solverFilename):
    solverSpec = readSolver(solverFilename)
    if not solverSpec.HasField(MODEL_FIELD):
        raise Exception('solver.prototxt provides no caffe model!')
    return solverSpec.net


def readSolver(filename):
    solverSpec = caffe_pb2.SolverParameter()
    readFromPrototxt(solverSpec, filename)
    return solverSpec


def writeToPrototxt(message, filename):
    with open(filename, 'w') as f:
        msgStr = google.protobuf.text_format.MessageToString(message)
        f.write(msgStr)


def readFromPrototxt(message, filename):
    with open(filename, 'r') as f:
        google.protobuf.text_format.Merge(f.read(), message)

