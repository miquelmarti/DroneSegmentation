#!/bin/bash

LOG_FILENAME=logs/job2_$(date "+%F-%T").log
~/hgRepos/transferLearningFramework/scripts/transfer.py --gpu $1 2_fcn8_cocopascal.prototxt 2>&1 | tee $LOG_FILENAME
