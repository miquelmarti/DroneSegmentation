#!/bin/bash

LOG_FILENAME=logs/job3_$(date "+%F-%T").log
~/hgRepos/transferLearningFramework/scripts/transfer.py --gpu $1 3_fcnresnet_coco_pascal.prototxt --clean 2>&1 | tee $LOG_FILENAME
