#!/bin/bash
# Expects bash get_losses.sh /path/to/source/info.log /path/to/destination
# By the way, you can save caffe log during the training with $(train_command) 2>&1 | tee info.log
# bash /path/to/info.log /path/to/loss.txt


# Get arguments
src=$1
dst=$2

cat $src | grep loss > 'loss_iter.txt'
cat 'loss_iter.txt' | grep Iteration > 'loss_line.txt'
cut -d ' ' -f 9 'loss_line.txt' > $dst

rm 'loss_iter.txt'
rm 'loss_line.txt'


