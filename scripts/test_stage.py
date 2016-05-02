#!/usr/bin/env python

# import numpy as np
import unittest
import os
from caffe.proto import caffe_pb2
import stage
import caffeUtils
import urllib2

TEST_DIR = '../test_cases'
TEST_NET_FILE = 'train.prototxt'
TEST_MODEL_URL_FILE = 'caffemodel-url'
TEST_MODEL_FILE = 'fcn8s-heavy-pascal.caffemodel'


class TestIgnore(unittest.TestCase):
    def setUp(self):
        caffeUrlPath = os.path.join(TEST_DIR, TEST_MODEL_URL_FILE)
        with open(caffeUrlPath, 'r') as urlFile:
            url = urlFile.read()
            modelName = os.path.basename(url)
            modelPath = os.path.join(TEST_DIR, modelName)
            if not os.path.isfile(modelPath):
                print 'Downloading test caffemodel from', url
                response = urllib2.urlopen(url)
                with open(modelPath, 'w') as f:
                    f.write(response.read())
                print 'Download of', modelPath, 'complete'

    def tearDown(self):
        # abusing the tearDown method here to avoid code duplication
        modelFilePath = os.path.join(TEST_DIR, TEST_MODEL_FILE)
        outModel = stage.ignoreModelLayers(self.ignoreLayers, modelFilePath)
        self.helper_assertIgnored(self.ignoreLayers, modelFilePath, outModel)
        os.remove(outModel)

    def helper_assertIgnored(self, ignoreLayers, oldModelFile, newModelFile):
        oldModel = caffeUtils.readCaffeModel(oldModelFile)
        oldLayerNames = [layer.name for layer in oldModel.layer]
        newModel = caffeUtils.readCaffeModel(newModelFile)
        newLayerNames = [layer.name for layer in newModel.layer]
        for name in oldLayerNames:
            if name in ignoreLayers:
                self.assertNotIn(name, newLayerNames)
            else:
                self.assertIn(name, newLayerNames)

    # TODO write test methods
    # ignore one convolutional layer
    def test_ignoreConvLayer(self):
        self.ignoreLayers = ['conv3_2']

    # ignore one relu layer
    def test_ignoreReluLayer(self):
        self.ignoreLayers = ['relu2_1']

    # ignore one softmax loss layer
    def test_ignoreLossLayer(self):
        self.ignoreLayers = ['loss']
        
    # ignore multiple layers of various types
    def test_ignoreVariousLayers(self):
        self.ignoreLayers = ['conv1_1', 'conv1_1', 'relu2_1', 'conv3_3',
                             'relu1_2', 'relu5_1']

    # ignore no layers
    def test_ignoreNoLayers(self):
        self.ignoreLayers = []


class TestFreeze(unittest.TestCase):

    def tearDown(self):
        # abusing the tearDown method here to avoid code duplication
        testNetPath = os.path.join(TEST_DIR, TEST_NET_FILE)
        outFile = stage.freezeNetworkLayers(self.freezeLayers, testNetPath)
        self.helper_assertFrozen(self.freezeLayers, outFile)
        os.remove(outFile)

    def helper_assertFrozen(self, freezeLayers, netFile):
        net = caffe_pb2.NetParameter()
        caffeUtils.readFromPrototxt(net, netFile)
        for layer in net.layer:
            if layer.name in freezeLayers:
                self.assertTrue(len(layer.param) > 0)
                for p in layer.param:
                    self.assertTrue(p.HasField(stage.LR_MULT_FIELD))
                    self.assertEqual(p.lr_mult, 0.0)

    # freezing a single layer with an lr_mult already defined
    def test_freezeSingleMult(self):
        self.freezeLayers = ['conv1_1']
        
    # freezing a single layer with two lr_mults already defined
    def test_freezeTwoMults(self):
        self.freezeLayers = ['conv2_1']

    # freezing a single layer with no lr_mult defined
    def test_freezeNoMult(self):
        self.freezeLayers = ['conv3_1']
    
    # freezing a single layer with lr_mult=0 already (nothing should happen)
    def test_freezeZeroMult(self):
        self.freezeLayers = ['conv4_1']

    # freezing multiple layers with all of the above properties
    def test_multiFreeze(self):
        self.freezeLayers = ['conv1_1', 'conv1_2',  # single lr_mult
                             'conv2_2',  # two lr_mults
                             'conv3_1', 'conv3_3',  # no lr_mult
                             'conv4_2', 'conv4_3']  # zero-valued lr_mult

if __name__ == '__main__':
    for case in [TestFreeze, TestIgnore]:
        suite = unittest.TestLoader().loadTestsFromTestCase(case)
        unittest.TextTestRunner(verbosity=2).run(suite)
    
