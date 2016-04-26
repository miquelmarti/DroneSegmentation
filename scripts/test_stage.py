#!/usr/bin/env python

# import numpy as np
import unittest
import os
from caffe.proto import caffe_pb2
# import transfer
import stage
import caffeUtils

TEST_NET_FILE = '../test_cases/train.prototxt'
TEST_MODEL_FILE = ''  # TODO fill this in


class TestIgnore(unittest.TestCase):

    def tearDown(self):
        # abusing the tearDown method here to avoid code duplication
        outModel = stage.ignoreModelLayers(self.ignoreLayers, TEST_MODEL_FILE)
        self.helper_assetIgnored(self.ignoreLayers, TEST_MODEL_FILE, outModel)
        os.remove(outModel)

    def helper_assertIgnored(self, ignoreLayers, oldModelFile, newModelFile):
        oldModel = caffeUtils.readCaffeModel(oldModelFile)
        oldLayerNames = [layer.name for layer in oldModel.layer]
        newModel = caffeUtils.readCaffeModel(oldModelFile)
        newLayerNames = [layer.name for layer in newModel.layer]
        for name in oldLayerNames:
            if name in ignoreLayers:
                self.assertNotIn(name, newLayerNames)
            else:
                self.assertIn(name, newLayerNames)


class TestFreeze(unittest.TestCase):

    def tearDown(self):
        # abusing the tearDown method here to avoid code duplication
        outFile = stage.freezeNetworkLayers(self.freezeLayers, TEST_NET_FILE)
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
    suite = unittest.TestSuite([TestIgnore(), TestFreeze()])
    # suite = unittest.TestLoader().loadTestsFromTestCase(TestFreeze)
    # unittest.TextTestRunner(verbosity=2).run(suite)
    suite.run()
    
