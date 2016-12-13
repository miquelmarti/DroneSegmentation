#!/bin/bash

# Needs relative path to sub-folder containing dataset with images and ground_truth folders
# E.g. ./createlist.sh hikawaES/

find $PWD/$1images -type f > tmp
find $PWD/$1ground_truth -type f > tmp2
paste -d " " tmp tmp2 > ./$1/list.txt
rm tmp tmp2
