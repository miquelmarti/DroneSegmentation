#!/usr/bin/env python

# import numpy as np
import unittest
import os
from caffe.proto import caffe_pb2
# import transfer
import stage
import caffeUtils

TEST_NET_FILE = '../test_cases/train.prototxt'


class TestFreeze(unittest.TestCase):

    def setUp(self):
        self.outFile = None

    def tearDown(self):
        # abusing the tearDown method here to avoid code duplication
        outFile = stage.freezeNetworkLayers(self.freezeLayers, TEST_NET_FILE)
        self.helper_assertFrozen(self.freezeLayers, outFile)
        if self.outFile:
            os.remove(self.outFile)
        self.outFile = None

    def helper_assertFrozen(self, freezeLayers, netFile):
        net = caffe_pb2.NetParameter()
        caffeUtils.readFromPrototxt(net, netFile)
        for layer in net.layer:
            if layer.name in freezeLayers:
                self.assertTrue(len(layer.param) > 0)
                for p in layer.param:
                    self.assertTrue(p.HasField(stage.LR_MULT_FIELD))
                    self.assertEqual(p.lr_mult, 0)

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
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFreeze)
    unittest.TextTestRunner(verbosity=2).run(suite)
