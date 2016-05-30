#!/bin/bash

LOG_FILENAME=logs/job4_$(date "+%F-%T").log
~/transferLearningFramework/scripts/transfer.py --gpu $1 4_fcnresnet_cocopascal.prototxt --clean 2>&1 | tee $LOG_FILENAME
