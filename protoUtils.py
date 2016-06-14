"""A collection of utility functions for protocol buffers and caffe."""

# TODO: Should be flexible to other way to specify train / test
# TODO: (see caffe proto)

import google.protobuf
from caffe.proto import caffe_pb2
import os

NET_FIELD = 'net'
TRAIN_NET_FIELD = 'train_net'


def replaceTrainNetFilename(solverSpec, newTrainNet):
    if solverSpec.HasField(NET_FIELD):
        solverSpec.net = newTrainNet
    elif solverSpec.HasField(TRAIN_NET_FIELD):
        solverSpec.train_net = newTrainNet
    else:
        raise Exception('No training network provided.')
    return solverSpec


def getTrainNetFilename(solverFilename):
    solverSpec = readSolver(solverFilename)
    if solverSpec.HasField(NET_FIELD):
        return solverSpec.net
    elif solverSpec.HasField(TRAIN_NET_FIELD):
        return solverSpec.train_net
    else:
        raise Exception(solverFilename + ' provides no training network!')


def getTrainNetFilenameFromProto(solverSpec):
    if solverSpec.HasField(NET_FIELD):
        return solverSpec.net
    elif solverSpec.HasField(TRAIN_NET_FIELD):
        return solverSpec.train_net
    else:
        raise Exception(solverFilename + ' provides no training network!')


def readSolver(filename):
    solverSpec = caffe_pb2.SolverParameter()
    readFromPrototxt(solverSpec, filename)
    return solverSpec


def writeToPrototxt(message, filename):
    with open(filename, 'w') as f:
        msgStr = google.protobuf.text_format.MessageToString(message)
        f.write(msgStr)


def readFromPrototxt(message, filename):
    filename = os.path.abspath(filename)
    with open(filename, 'r') as f:
        google.protobuf.text_format.Merge(f.read(), message)


def readCaffeModel(modelFilename):
    with open(modelFilename, 'r') as f:
        net = caffe_pb2.NetParameter()
        net.ParseFromString(f.read())
        return net


def getMeanFromBinaryproto(meanFilename):
    with open(meanFilename, 'r') as f:
        meanBlob = caffe_pb2.BlobProto()
        meanBlob.ParseFromString(f.read())
