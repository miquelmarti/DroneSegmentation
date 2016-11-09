#!/bin/bash

LOG_FILENAME=logs/job1_$(date "+%F-%T").log
~/transferLearningFramework/scripts/transfer.py --gpu $1 1_fcn8_coco_pascal.prototxt 2>&1 | tee $LOG_FILENAME
