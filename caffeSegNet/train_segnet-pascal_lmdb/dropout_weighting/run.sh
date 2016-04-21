#!/bin/sh
export PYTHONPATH="/home/shared/caffeSegNet/python"
python /home/pierre/hgRepos/caffeTools/runSegmentation.py --model /home/pierre/hgRepos/models/caffeSegNet/train_segnet-pascal_lmdb/dropout_weighting/deploy.prototxt --weights /home/pierre/hgRepos/models/caffeSegNet/train_segnet-pascal_lmdb/dropout_weighting/train/train_iter_18000.caffemodel --colours /home/shared/datasets/VOCdevkit/VOC2012/colors/pascal_voc_21_colors.png --output prob --labels /home/shared/datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/val_img_lab.txt
