'''
RUN COMMAND EXAMPLE :
python view_weights.py --model /home/shared/caffeFcn/models/fcn-8s-pascal/deploy.prototxt --weights /home/shared/caffeFcn/models/fcn-8s-pascal/fcn-8s-pascal.caffemodel --layer "conv1_1"
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image
import argparse
import time
import cv2
import os


# Caffe module need to be on python path
import caffe

def get_arguments():
    # Import arguments
    parser = argparse.ArgumentParser()
    
    # Mandatory options
    parser.add_argument('--model', type=str, required=True, help='Path to the model (usually [...]/deploy.prototxt)')
    parser.add_argument('--weights', type=str, required=True, help='Path to the weights (usually [...]/xx.caffemodel)')
    parser.add_argument('--layer', type=str, required=True, help='Name of the layer to visualise')
    
    # Optional options
    parser.add_argument('--gpu', type=bool, required=False, default=True, help='Default true, set False for CPU mode')
    
    return parser.parse_args()


def build_network(args):
    # GPU / CPU mode
    if args.gpu == False:
        print 'Set CPU mode'
        caffe.set_mode_cpu()
    else:
        print 'Set GPU mode'
        caffe.set_mode_gpu()
    
    # Creation of the network
        net = caffe.Net(args.model,      # defines the structure of the model
                        args.weights,    # contains the trained weights
                        caffe.TEST)      # use test mode (e.g., don't perform dropout)
    return net



if __name__ == '__main__':

    # Get all options
    args = get_arguments()
    
    # Set the network according to the arguments
    net = build_network(args)

    # Which layer to see ?
    layer_to_see = args.layer

    # Extract weights
    weights = net.params[layer_to_see][0].data
    print "WEIGHTS of layer", layer_to_see, " (of shape ", weights.shape ,") : ", weights

    # Extract bias
    bias = net.params[layer_to_see][1].data
    print "BIAS (of shape ", bias.shape ,") : ", bias

    # Show image of the weights
    filter_number_to_show = 0
    channel_to_show = 0
    image_to_show = weights[filter_number_to_show,channel_to_show,:,:]
    cv2.imshow("Weights", cv2.resize(image_to_show, (150, 150)) )
    cv2.waitKey(0)
