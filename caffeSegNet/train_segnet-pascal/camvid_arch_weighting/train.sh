#!/bin/sh
export PYTHONPATH="/home/shared/caffeSegNet/python"
/home/shared/caffeSegNet/build/tools/caffe train --solver="/home/pierre/hgRepos/models/caffeSegNet/train_segnet-pascal/camvid_arch_weighting/solver.prototxt" 2>&1 | tee /home/pierre/hgRepos/models/caffeSegNet/train_segnet-pascal/camvid_arch_weighting/train/info.log
