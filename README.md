## FOLDERS ########################
# create dataset folder
Contains many files for dealing with the providing datasets and convert them into a readable format for caffe

## FILES ##########################

# trainNetwork.py
A script for training caffe networks.  This replaces the ‘solve.py’ scripts used by most of FCN architectures from the original Berkeley github repos.

When using this, be aware that many of the values in specified in Caffe's solver.prototxt that are ignored by the original solve.py files are *not* ignored by trainNetwork.py.  test_interval, test_iter and max_iter are the most notable examples of this.  Be sure that you have set these properly in your solver.prototxt file.  Also please be aware of the behaviour of your chosen data layer - the statistics computed during the test phase will depend on its behaviour, particularly if your test_iter value is bigger than the number of samples in your test set.

# segment.py
A script for testing semantic-segmentation networks, both individually and as ensembles, by running them on a provided dataset and computing metrics on the results.  As input it takes a .prototxt config file - the format of this config file is defined in ensemble.proto, and an example is provided in test_example.prototxt.

# display_loss.py
A script for plotting the loss, mean IU, and other statistics contained in the training logfiles recorded from Caffe’s output.

# get_class_weights.py
In SegNet, the loss layer weights each class by its frequency in the dataset.  This script computes the class frequencies of a dataset so that they can be used to run SegNet on those datasets.

# view_weights.py
Visualize the weights of a layer from the caffemodel of a given iteration.

# configure
A simple bash script that auto-generates protocol-buffer python files from .proto message definition files.  When you first check out this repo, this script will need to be run before you can use some of the other scripts that rely on it.  You’ll also need to run it again if you make changes to any .proto files in this repo.
