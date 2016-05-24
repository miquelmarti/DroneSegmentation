### CIFAR10
This example is based on the Caffe CIFAR10 training example (http://caffe.berkeleyvision.org/gathered/examples/cifar10.html).

-> Prerequisites
You will need to download the cifar10 dataset :
cd data
bash get_cifar10.sh

-> Run
python scripts/transfer.py examples/transfer_tests/transfer_cifar10.prototxt



### FLICKR STYLE (finetuning CaffeNet)
This example is based on the Caffe Flickr Style training example for style recognition (http://caffe.berkeleyvision.org/gathered/examples/finetune_flickr_style.html).

-> Prerequisites
You will need to download the flickr_style dataset :
cd data
bash get_flickr_style.sh

And also the weights and mean of the source model (bvlc_reference_caffenet, trained on ImageNet) :
cd $CAFFE_ROOT
bash data/ilsvrc12/get_ilsvrc_aux.sh

-> Run
python scripts/transfer.py examples/transfer_tests/transfer_flickr.prototxt --model /home/shared/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel



### FCN
This example is based on the FCN models - FCN32, FCN16 and FCN8 (https://github.com/shelhamer/fcn.berkeleyvision.org).

-> Prerequisites
You will need to download Pascal VOC 2012
You will need to download SBD dataset

-> Run
python scripts/transfer.py examples/transfer_tests/transfer_fcn.prototxt --model /home/shared/givenModels/vgg-16/VGG_ILSVRC_16_layers.caffemodel



### MULTI-SOURCES 
Multi-Source implementation, based on the FCN models (https://github.com/shelhamer/fcn.berkeleyvision.org).

# Prerequisites
You will need to download Pascal VOC 2012
You will need to download SBD dataset

# Run
python scripts/transfer.py examples/transfer_tests/one_stage_five_iter_ms.prototxt --model /home/shared/givenModels/fcn-16s-pascal/fcn-16s-pascal.caffemodel
python scripts/transfer.py examples/transfer_tests/one_stage_one_iter_ms.prototxt --model /home/shared/givenModels/fcn-16s-pascal/fcn-16s-pascal.caffemodel
python scripts/transfer.py examples/transfer_tests/three_stage_five_iter_ms.prototxt --model /home/shared/givenModels/vgg-16/VGG_ILSVRC_16_layers.caffemodel
python scripts/transfer.py examples/transfer_tests/two_stage_five_iter_ms.prototxt --model /home/shared/givenModels/fcn-32s-pascal/fcn-32s-pascal.caffemodel
python scripts/transfer.py examples/transfer_tests/two_stage_one_iter_ms.prototxt --model /home/shared/givenModels/fcn-32s-pascal/fcn-32s-pascal.caffemodel
python scripts/transfer.py examples/transfer_tests/zero_iter_ms.prototxt --model /home/shared/givenModels/vgg-16/VGG_ILSVRC_16_layers.caffemodel


