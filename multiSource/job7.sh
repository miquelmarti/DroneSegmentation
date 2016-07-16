#!/bin/bash

LOG_FILENAME=logs/job7_$(date "+%F-%T").log
~/transferLearningFramework/scripts/transfer.py --gpu $1 7_fcnresnetskip_coco_pascal_atonce.prototxt 2>&1 | tee $LOG_FILENAME
