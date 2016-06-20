# TODO: 

import os
import caffe
from caffe.proto import caffe_pb2
import transferLearning_pb2
from caffeUtils import protoUtils, solve
import logging
logger = logging.getLogger()

# names of parameters in protobufs so we can check for their presence
SNAP_PREFIX = '[snapshot_info]'


class Snapshot(object):
    """
    A class that represents a transfer-learning snapshot.
    """

    def __init__(self, multisource=-1, iteration=-1, best_weights=[],
                 best_scores=[], stage=None, stage_weights=None,
                 stage_snapshot=None, out_filename=''):
        """
        Constructor for the Snapshot class.

        Arguments:
        multisource -- The current multisource to execute
        iteration -- The current iteration of this multisource
        """

        self.multisource = multisource
        self.iteration = iteration
        self.best_weights = best_weights
        self.best_scores = best_scores
        self.stage = stage
        self.stage_weights = stage_weights
        self.stage_snapshot = stage_snapshot
        self.out_filename = out_filename
        
        self.restored = False

    def verify(self):
        assert self.multisource >= 0, \
        ' '.join["'multisource' should be positive, here:", self.multiSource]
        assert self.iteration >= 0, \
        ' '.join["'iteration' should be positive, here:", self.iteration]
        assert len(self.best_weights) >= 0, \
        ' '.join["Should provide at least one best_weights, even empty"]
        assert len(self.best_scores) >= 0, \
        ' '.join["Should provide at least one best_scores, even empty"]
        assert len(self.best_weights) == len(self.best_scores), \
        ' '.join["The number of best_weights should be equal to the number",
                 "of best_scores"]
        assert self.stage >= 0, \
        ' '.join["'stage' should be positive, here:", self.multiSource]
    
    def snapToMsg(self, snap):
        msg = transferLearning_pb2.Snapshot()
        msg.multisource = snap.multisource
        msg.iteration = snap.iteration
        for i in range(len(snap.best_weights)):
            msg.best_weights.append(snap.best_weights[i])
            msg.best_scores.append(snap.best_scores[i])
        
        if not snap.stage is None:
            msg.stage = snap.stage
        if not snap.stage_weights is None:
            msg.stage_weights = snap.stage_weights
        if not snap.stage_snapshot is None:
            msg.stage_snapshot = snap.stage_snapshot
        
        return msg
    
    def msgToSnap(self, msg):
        snap = Snapshot(msg.multisource, msg.iteration, msg.best_weights, 
                        msg.best_scores, msg.stage, msg.stage_weights,
                        msg.stage_snapshot)
        return snap
        

    def save(self, silent=False):
        snap_msg = self.snapToMsg(self)
        
        # Write the snapshot
        protoUtils.writeToPrototxt(snap_msg, self.out_filename)
        if not silent:
            logger.info('%s Write %s', SNAP_PREFIX, os.path.basename(self.out_filename))

    def copyFrom(self, fileName):
        msg = transferLearning_pb2.Snapshot()
        protoUtils.readFromPrototxt(msg, fileName)
        snap = self.msgToSnap(msg)
        return snap


