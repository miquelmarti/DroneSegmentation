#!/bin/sh
export PYTHONPATH="/home/shared/caffeSegNet/python"
/home/shared/caffeSegNet/build/tools/caffe train --solver="/home/pierre/hgRepos/models/caffeSegNet/train_segnet-pascal_lmdb/johannes_version/solver.prototxt" --weights="/home/pierre/tmpModels/VGG/VGG_ILSVRC_16_layers.caffemodel" 2>&1 | tee /home/pierre/hgRepos/models/caffeSegNet/train_segnet-pascal_lmdb/johannes_version/train/info.log
