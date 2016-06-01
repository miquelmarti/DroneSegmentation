#!/bin/bash

LOG_FILENAME=logs/job3_$(date "+%F-%T").log
~/transferLearningFramework/scripts/transfer.py --gpu $1 3_fcnresnet_coco_pascal.prototxt 2>&1 | tee $LOG_FILENAME
