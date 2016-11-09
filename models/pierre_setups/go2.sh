#!/bin/bash

name=$1
if [ ${name: -9} == "*.prototxt" ]; then
    name=$(echo $name | cut -f 1 -d '.')
fi
if [ -z "$2" ]; then
    gpu=0
else
    gpu=$2
fi

file_prototxt=$name.prototxt
dir_results=results/$name
file_log=$dir_results/$name.log

if [ ! -f $file_prototxt ]; then
    echo "File $file_prototxt not found!"
    exit
fi

if [[ ! -e results ]]; then
    mkdir results
fi
if [[ ! -e $dir_results ]]; then
    mkdir $dir_results
fi

/home/johannes/transferLearningFramework/scripts/transfer.py --gpu $gpu $file_prototxt 2>&1 | tee $file_log


