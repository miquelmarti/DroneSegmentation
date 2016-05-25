#!/bin/bash

LOG_FILENAME=logs/job2_$(date "+%F-%T").log
~/transferLearningFramework/scripts/transfer.py --gpu $1 2_fcn8_cocopascal.prototxt --clean 2>&1 | tee $LOG_FILENAME
