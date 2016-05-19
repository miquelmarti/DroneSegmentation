CIFAR10
- merge the solve.py with the one for FCN

FLICKR_STYLE
- the path to the weights (in transfer.prototxt) are absolute (in caffe_root) is it ok ?
- idem for imagenet_mean.bynaryproto (in train_val.prototxt)

FCN
- Correct weights problem : for FCN32, we give weights in the command, so these are not modified by python (no ignore or freeze). If we give the weights as arguments for transfer.py, they are not transferred as argument to the command line
- The solve.py is pretty simmilar to the one in the cifar10 directory. Find a way to put them both as one file
- Deal with VOC / SBD dataset download
- The link to VGG16 in transfer.prototxt is absolute
- doesn't work without layers.py
- Do something for the max_iter problem (check((max_iter > snapshot) && (max_iter % snapshot == 0))

MULTI-SOURCE
- the validation set points to Pascal VOC, absolute link, for all the datasets (cifar, etc)
- deal with other scores than mean IU for classification (and not pixel-wise segmentation)
- be more flexible with outLayer, find a way to get it from the trainNet
- get the weights (for FCN) from absolute paths
