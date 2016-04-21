#!/bin/sh
export PYTHONPATH="/home/shared/caffeSegNet/python"
python /home/pierre/hgRepos/caffeTools/runSegmentation.py --model /home/pierre/hgRepos/models/caffeSegNet/train_segnet-pascal_lmdb/johannes_version/deploy.prototxt --weights /home/pierre/hgRepos/models/caffeSegNet/train_segnet-pascal_lmdb/johannes_version/train/train_iter_24000.caffemodel --colours /home/shared/datasets/VOCdevkit/VOC2012/colors/pascal_voc_21_colors_2.png --output prob --labels /home/shared/datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/val_img_lab.txt
