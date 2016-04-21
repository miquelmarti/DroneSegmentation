#!/bin/sh
export PYTHONPATH="/home/shared/caffeSegNet/python"
python /home/pierre/hgRepos/caffeTools/runSegmentation.py --model /home/pierre/hgRepos/models/caffeSegNet/train_segnet-pascal_lmdb/camvid_arch_no_weighting/deploy.prototxt --weights /home/pierre/hgRepos/models/caffeSegNet/train_segnet-pascal_lmdb/camvid_arch_no_weighting/train/train_iter_14000.caffemodel --colours /home/shared/datasets/VOCdevkit/VOC2012/colours/pascal_voc_21_colours.png --output prob --labels /home/shared/datasets/VOCdevkit/VOC2012/lists/val_img_lab.txt
