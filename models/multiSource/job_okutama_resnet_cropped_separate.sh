#!/bin/bash

LOG_FILENAME=logs/job_okutama_resnet_cropped_separate_$(date "+%F-%T").log
~/hgRepos/transferLearningFramework/scripts/transfer.py --gpu $1 okutama_resnet_cropped_separate.prototxt 2>&1 | tee $LOG_FILENAME
