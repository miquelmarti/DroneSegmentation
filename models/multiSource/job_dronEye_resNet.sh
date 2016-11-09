#!/bin/bash

LOG_FILENAME=logs/job_dronEye_resNet_$(date "+%F-%T").log
~/hgRepos/transferLearningFramework/scripts/transfer.py --gpu $1 dronEye_resNet_100000.prototxt 2>&1 | tee $LOG_FILENAME
