#!/bin/sh

MODEL_PATH="/home/pierre/hgRepos/models/caffeSegNet/train_segnet-pascal_lmdb/dropout_no_weighting"

export PYTHONPATH="/home/shared/caffeSegNet/python"
/home/shared/caffeSegNet/build/tools/caffe train            \
                --solver    ${MODEL_PATH}/solver.prototxt   \
                2>&1 | tee  ${MODEL_PATH}/train/info.log
