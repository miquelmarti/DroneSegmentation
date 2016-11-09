# TODO: Check if we can get the repeated field without for

import os
import caffe
from caffe.proto import caffe_pb2
import transferLearning_pb2
from caffeUtils import protoUtils, solve

# Prefixes
P_SOLVER = '[solverParam_info]'

# Extensions
E_TEMP_FILE = '.tmp'

# Fields
F_NET       = 'net'
F_TRAIN_NET = 'train_net'
F_MAX_ITER  = 'max_iter'
F_TEST_INTERVAL = 'test_interval'
F_SNAPSHOT = 'snapshot'
F_SNAPSHOT_PREFIX = 'snapshot_prefix'


class SolverParam(object):
    """
    A class that represents a transfer-learning solver parameters.
    """

    def __init__(self, filename):
        """
        Constructor for the SolverParam class.

        Arguments:
        filename -- Absolute path to the solver
        """
        
        # Absolute path to the solver
        self.filename = filename
        self.directory = os.path.dirname(filename)
        # Prototxt with all the solver charcteristics
        self.solverSpec = protoUtils.readSolver(str(self.filename))
        
        # To initialize
        self.trainNetwork = None
        self.testNetworks = []
        self.maxIter = 0
        self.testIter = []
        self.testInterval = 0
        self.snapshotInterval = 0
        self.snapshotPrefix = None
        
        # Get absolute path to train network
        if self.solverSpec.HasField(F_NET):
            self.trainNetwork = os.path.join(self.directory, 
                                             self.solverSpec.net)
        elif self.solverSpec.HasField(F_TRAIN_NET):
            self.trainNetwork = os.path.join(self.directory,
                                             self.solverSpec.train_net)
        
        # Get test networks
        if len(self.solverSpec.test_net) > 0:
            for i in range(len(self.solverSpec.test_net)):
                self.testNetworks.append(os.path.join(self.directory,
                                         self.solverSpec.test_net[i]))
        if len(self.solverSpec.test_net_param) > 0:
            print "'test_net_param' not handle by this framework, use test_net."
        
        # Get number of iterations for test
        if self.solverSpec.HasField(F_MAX_ITER):
            self.maxIter = self.solverSpec.max_iter
        
        # Get max number of iterations
        if len(self.solverSpec.test_iter) > 0:
            for i in range(len(self.solverSpec.test_iter)):
                self.testIter.append(self.solverSpec.test_iter[i])
        
        # Get test interval
        if self.solverSpec.HasField(F_TEST_INTERVAL):
            self.testInterval = self.solverSpec.test_interval
        
        # Get snapshot interval
        if self.solverSpec.HasField(F_SNAPSHOT):
            self.snapshotInterval = self.solverSpec.snapshot
        
        # Get snapshot prefix
        if self.solverSpec.HasField(F_SNAPSHOT_PREFIX):
            self.snapshotPrefix = os.path.join(self.directory, 
                                               self.solverSpec.snapshot_prefix)
            self.snapshotPrefix += '_iter_'
    
    
    def verify(self):
        # TODO: Deal with net_param and train_net_param
        assert self.trainNetwork, \
        ' '.join(["No network provided. If you used 'net_parm' or", \
                 "'train_net_param' to initialize the network, please use", \
                 "'net' or 'train_net'."])
        
        assert os.path.isfile(self.trainNetwork), \
        ' '.join(["The given train network (", self.trainNetwork, ")", \
                  "doesn't exist"])
        
        if len(self.testNetworks) > 0:
            for t in self.testNetworks:
                assert os.path.isfile(t), \
                ' '.join(["The given test network (", t, ") doesn't exist"])
            
            assert len(self.testNetworks) == len(self.testIter), \
            ' '.join(["The solver should provide as much test networks as", \
                      "test iterations. Here (", len(self.testNetworks), \
                      "vs", len(self.testIter), ")"])
        
        assert self.maxIter > 0, \
        ' '.join(["Max Iter has to be positive (and != 0). Here:", \
                  self.maxIter])
        assert self.testInterval >= 0, \
        ' '.join(["Test Interval has to be positive. Here:", \
                  self.testInterval])
        assert self.snapshotInterval >= 0, \
        ' '.join(["Snapshot Interval has to be positive. Here:", \
                  self.snapshotInterval])
        
        if self.snapshotInterval > 0:
            assert self.snapshotPrefix, \
            ' '.join(["The solver should provide a snapshot prefix."])
    
    
    def replaceTrainNetwork(self, newNetwork):
        if self.solverSpec.HasField(NET_FIELD):
            self.solverSpec.net = newNetwork
        elif self.solverSpec.HasField(TRAIN_NET_FIELD):
            self.solverSpec.train_net = newNetwork
        
        self.trainNetwork = newNetwork
        self.filename += E_TEMP_FILE
        protoUtils.writeToPrototxt(self.solverSpec, self.filename)
        




