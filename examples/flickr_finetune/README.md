## Finetuning CaffeNet for style recognition on "Flickr Style" data
=> From http://caffe.berkeleyvision.org/gathered/examples/finetune_flickr_style.html


# Datasets
You will need to download the flickr_style dataset :
cd data
bash get_flickr_style.sh

And also the weights and mean of the source model (bvlc_reference_caffenet, trained on ImageNet) :
cd $CAFFE_ROOT
bash data/ilsvrc12/get_ilsvrc_aux.sh


# Run
Command for the finetuning :
python scripts/transfer.py examples/prototxts/transfer_flickr.prototxt --model /home/shared/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel


