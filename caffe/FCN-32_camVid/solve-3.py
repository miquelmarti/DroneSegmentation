import caffe
import surgery
import score

import numpy as np
import os
import sys

import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))

weights = 'fcn32s-heavy-pascal.caffemodel'

# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver-3.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
#val = np.loadtxt('/home/shared/datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/seg11valid.txt', dtype=str)

for _ in range(25):
    solver.step(4000)
    #score.seg_tests(solver, False, val, layer='score')
