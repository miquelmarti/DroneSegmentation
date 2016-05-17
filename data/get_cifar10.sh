#!/usr/bin/env bash
# This scripts downloads the CIFAR10 (binary version) data and converts it to
# LMDB format.  You may need to run this script as root (hint: use sudo).

CAFFE=/home/shared/caffe
CIFAR10_DIR=examples/cifar10
DATASETS_CIFAR=data/cifar10

if  ! test -d $CAFFE/$CIFAR10_DIR/cifar10_train_lmdb ||
        ! test -d $CAFFE/$CIFAR10_DIR/cifar10_train_lmdb; then
    cd $CAFFE
    ./$DATASETS_CIFAR/get_cifar10.sh
    $CIFAR10_DIR/create_cifar10.sh
    cd -
fi

if [ ! -d $DATASETS_CIFAR ] 
then
    mkdir -p $DATASETS_CIFAR
fi

cp -r $CAFFE/$CIFAR10_DIR/cifar10_*_lmdb $DATASETS_CIFAR
cp -r $CAFFE/$CIFAR10_DIR/mean.binaryproto $DATASETS_CIFAR
