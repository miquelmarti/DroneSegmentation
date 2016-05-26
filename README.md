## FOLDERS ########################
# create dataset folder
Contains many files for dealing with the providing datasets and convert them into a readable format for caffe

## FILES ##########################
# classify.py
[...]

# display_loss.py
Take the info.log as input and display the training loss and other curves

# get_class_weights.py
In SegNet, the loss layer is weighting each classes. These one are given for CamVid, this code permits to get them for other datasets.

# runSegmentation.py
Used for the visualization.

# view_weights.py
Visualize the weights of a layer from the caffemodel of a given iteration.

We have some changes to permissions...
