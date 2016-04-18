#!/bin/sh
export PYTHONPATH="/home/shared/caffeSegNet/python"
python /home/pierre/hgRepos/caffeTools/runSegmentation.py --model /home/pierre/hgRepos/models/caffeSegNet/train_segnet-camvid/deploy.prototxt --weights /home/pierre/hgRepos/models/caffeSegNet/train_segnet-camvid/train/train_iter_80000.caffemodel --colours /home/shared/datasets/CamVid/colours/camvid12.png --output prob --labels /home/shared/datasets/CamVid/test.txt
