#!/bin/sh
export PYTHONPATH="/home/shared/caffeSegNet/python"
/home/shared/caffeSegNet/build/tools/caffe train --solver="/home/pierre/hgRepos/models/caffeSegNet/train_segnet-camvid/solver.prototxt" 2>&1 | tee /home/pierre/hgRepos/models/caffeSegNet/train_segnet-camvid/train/info.log
