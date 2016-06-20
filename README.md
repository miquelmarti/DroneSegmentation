This module represents a collection of utility classes and functions used across several of our other repositories.

It is recommended that you add the directory of this repo to your PYTHONPATH environment variable.  For example, if you checked out this repo to /home/andrew, run the following command:

export PYTHONPATH=/home/andrew:$PYTHONPATH

You can also put the line:
PYTHONPATH=/home/andrew:$PYTHONPATH
in your .bashrc file, so that you won't have to run the export command every time you log in.

## FOLDERS ########################
# fcnLayers
this directory contains various python Caffe layers used to read in different types of data for training and testing.  See its own README.md for more information.

## FILES ##########################

# solve.py
This module contains methods for training and testing Caffe networks with Python.
# score.py
This module contains utility methods for computing the metrics we use to judge the quality of a semantic segmentation, such as pixel accuracy and mean IU.
# protoUtils.py
This module contains utility methods for interacting with protocol buffer objects.
