#!/bin/sh
export PYTHONPATH="/home/shared/caffeSegNet/python"
python /home/pierre/hgRepos/caffeTools/runSegmentation.py --model /home/pierre/hgRepos/models/caffeSegNet/train_segnet-camvid/deploy.prototxt --weights /home/pierre/hgRepos/models/caffeSegNet/train_segnet-camvid/train/train_iter_80000.caffemodel --colours /home/shared/datasets/CamVid/colours/camvid12.png --input data --output prob --labels /home/shared/datasets/CamVid/train.txt
