#!/bin/bash

LOG_FILENAME=logs/job4_$(date "+%F-%T").log
~/hgRepos/transferLearningFramework/scripts/transfer.py --gpu $1 4_fcnresnet_cocopascal.prototxt 2>&1 | tee $LOG_FILENAME
