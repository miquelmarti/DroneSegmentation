#!/usr/bin/env bash
# This scripts downloads the flickr_style dataset.  You may need to 
# run this script as root (hint: use sudo).

CAFFE=/home/shared/caffe
FLICKR_DIR=examples/finetune_flickr_style/
DATASETS_FLICKR=data/flickr_style

if ! test -d $CAFFE/$DATASETS_FLICKR; then
    cd $CAFFE
    $FLICKR_DIR/assemble_data.py --images=2000 --seed 29837
    cd -
fi

cp -r $CAFFE/data/flickr_style .
