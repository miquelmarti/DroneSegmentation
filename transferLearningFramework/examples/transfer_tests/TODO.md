FLICKR_STYLE
- The path to the weights (in transfer_cifar10.prototxt) are absolute (depending on caffe_root)
- Idem for imagenet_mean.bynaryproto (in cifar10/train_val_style.prototxt)

FCN
- Check for VOC / SBD dataset download
- The link to VGG16's weights in transfer_fcn.prototxt is absolute

MULTI-SOURCE
- The training   set has absolute paths to SBD
- The validation set has absolute paths to Pascal VOC
- Deal with other scores than mean IU, for classification per example (and not pixel-wise segmentation anymore)
- Be more flexible with outLayer, find a way to get it from the trainNet (not always "score" or "output")
- The link to FCN's real weights are absolute


