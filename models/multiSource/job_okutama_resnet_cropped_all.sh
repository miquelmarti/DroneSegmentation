#!/bin/bash

LOG_FILENAME=logs/job_okutama_resnet_cropped_all_$(date "+%F-%T")_2.log
~/hgRepos/transferLearningFramework/scripts/transfer.py --gpu $1 okutama_resnet_cropped_all.prototxt --resume /home/pierre/hgRepos/models/multiSource/okutama_resnet_cropped_all/okutama_resnet_cropped_all_snapshot.prototxt 2>&1 | tee $LOG_FILENAME
