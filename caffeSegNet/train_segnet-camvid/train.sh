#!/bin/sh
export PYTHONPATH="/home/shared/caffeSegNet/python"
/home/shared/caffeSegNet/build/tools/caffe train --solver="/home/pierre/hgRepos/models/caffeSegNet/train_segnet-camvid/solver.prototxt" --snapshot="/home/pierre/hgRepos/models/caffeSegNet/train_segnet-camvid/train/train_iter_10000.solverstate" 2>&1 | tee /home/pierre/hgRepos/models/caffeSegNet/train_segnet-camvid/train/info2.log
