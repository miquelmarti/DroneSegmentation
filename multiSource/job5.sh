#!/bin/bash

OUTPUT_DIR="job5Weights"
EXT_DIR="iter_"

# For each iteration
for iter in {1..10}; do
    LOG_FILENAME=logs/job5_$(date "+%F-%T")_$iter.log
    ~/hgRepos/transferLearningFramework/scripts/transfer.py --gpu $1 5_fcnresnet_pascal.prototxt --out_dir $OUTPUT_DIR/$EXT_DIR$iter 2>&1 | tee $LOG_FILENAME 
done
