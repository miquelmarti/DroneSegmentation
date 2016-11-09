#!/bin/sh

MODEL_PATH="/home/pierre/hgRepos/models/caffeSegNet/train_segnet-camvid_lmdb"
DATASET_PATH="/home/shared/datasets/CamVid"

export PYTHONPATH="/home/shared/caffeSegNet/python"
python /home/pierre/hgRepos/caffeTools/runSegmentation.py                       \
                --model     ${MODEL_PATH}/deploy.prototxt                       \
                --weights   ${MODEL_PATH}/train/train_iter_80000.caffemodel     \
                --colours   ${DATASET_PATH}/colours/camvid12.png                \
                --output    prob                                                \
                --resize                                                        \
                --labels    ${DATASET_PATH}/val.txt
