#!/bin/sh
export PYTHONPATH="/home/shared/caffeSegNet/python"
/home/shared/caffeSegNet/build/tools/caffe train --solver="/home/pierre/hgRepos/models/caffeSegNet/train_segnet-pascal_lmdb/solver.prototxt" 2>&1 | tee /home/pierre/hgRepos/models/caffeSegNet/train_segnet-pascal_lmdb/train/info.log
