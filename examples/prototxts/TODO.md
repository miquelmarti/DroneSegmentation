CIFAR10
- Merge the solve.py with the one for FCN

FLICKR_STYLE
- The path to the weights (in transfer.prototxt) are absolute (in caffe_root) is it ok ?
- Idem for imagenet_mean.bynaryproto (in train_val.prototxt)

FCN
- Correct weights problem : for FCN32, we give weights in the command line, so they are not modified by the framework (ignore or freeze). If we give the weights as arguments for transfer.py, they are not transferred as argument to the command line.
- Deal with VOC / SBD dataset download
- The link to VGG16 in transfer.prototxt is absolute
- Doesn't work without layers.py (for now, it is in my PYTHONPATH)

MULTI-SOURCE
- The validation set has absolute paths to Pascal VOC
- Deal with other scores than mean IU, for classification per example (and not pixel-wise segmentation)
- Be more flexible with outLayer, find a way to get it from the trainNet
- Get the weights (for FCN) from absolute paths
