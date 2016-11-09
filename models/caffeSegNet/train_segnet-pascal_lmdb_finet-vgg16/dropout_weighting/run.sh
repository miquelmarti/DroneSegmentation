#!/bin/sh

MODEL_PATH="/home/pierre/hgRepos/models/caffeSegNet/train_segnet-pascal_lmdb_finet-vgg16/dropout_weighting"
DATASET_PATH="/home/shared/datasets/VOCdevkit/VOC2012"

export PYTHONPATH="/home/shared/caffeSegNet/python"
python /home/pierre/hgRepos/caffeTools/runSegmentation.py                           \
                --model     ${MODEL_PATH}/deploy.prototxt                           \
                --weights   ${MODEL_PATH}/train/train_iter_80000.caffemodel         \
                --colours   ${DATASET_PATH}/colours/pascal_voc_21_colours.png       \
                --output    prob                                                    \
                --PASCAL                                                            \
                --resize                                                            \
                --labels    ${DATASET_PATH}/lists/seg11_img_lab.txt
