#!/bin/sh

MODEL_PATH="/home/pierre/hgRepos/models/caffeSegNet/train_segnet-pascal_finet-vgg16/camvid_arch_no_weighting"
WEIGHTS_PATH="/home/shared/givenModels/vgg-16"

export PYTHONPATH="/home/shared/caffeSegNet/python"
/home/shared/caffeSegNet/build/tools/caffe train                                \
                --solver    ${MODEL_PATH}/solver.prototxt                       \
                --weights   ${WEIGHTS_PATH}/VGG_ILSVRC_16_layers.caffemodel     \
                2>&1 | tee  ${MODEL_PATH}/train/info.log
