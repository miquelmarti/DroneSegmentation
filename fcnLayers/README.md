These python caffe layers are provided, unmodified, from the Github repo located at https://github.com/shelhamer/fcn.berkeleyvision.org/.  They are provided so that we can train the fcn networks provided in that repo, which make use of these layers as their data layers, for some reason, rather than just doing things with text files listing the input images or lmdbs like a normal person would.

coco_layers.py is the exception - it is based on Shelhamer's code but heavily modified to reduce code duplication and allow new dataset types.
