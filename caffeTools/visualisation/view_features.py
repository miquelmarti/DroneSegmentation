#!/usr/bin/env python

# Code for visualizing the main look of the weights
# Ex : python view_weights.py --model /home/johannes/project/models/segnet/deploy-pascal.prototxt --weights /home/johannes/project/models/segnet/train_pascal/ --layers "conv1_1" --scale 0.1 --shift -0.05


import numpy as np
import argparse
import cv2
import caffe
from PIL import Image

dataLayer = 'data'
mean = np.array((104.456043956, 132.197802198, 126.175824176))

def get_arguments():
    # Import arguments
    parser = argparse.ArgumentParser()
    
    # Mandatory options
    parser.add_argument('--model', required=True, help='\
    Path to the .prototxt model')
    parser.add_argument('--weights', required=True, help='\
    Path to the .args.weights weights')
    parser.add_argument('--layers', nargs='*', help='\
    Names of the layers to visualise (the name as seen in the prototxt)')
    parser.add_argument('--input', type=str, help='\
    Input image')
    parser.add_argument('--filter_num', type=int, help='\
    Number of filter')
    
    # Optional options
    parser.add_argument('--cpu', action="store_true", help='\
    Default False, set True for CPU mode')
    parser.add_argument('--scale', type=float, default=1.0, help='\
    Divides the data value by this before plot')
    parser.add_argument('--shift', type=float, default=1.0, help='\
    Shifts the data value by this before plot')
    parser.add_argument('--autoscale', action="store_true", help='\
    Sets automatically the scale to better print the weights')
    parser.add_argument('--nokey', action="store_true", help='\
    Move through the weights quickly')
    parser.add_argument('--save', type=str, default='', help='\
    Saves png files in this folder')
    
    return parser.parse_args()



def preProcessing(image, mean=None):
    # Get pixel values and convert them from RGB to BGR
    image = np.array(image, dtype=np.float32)
    if len(image.shape) is 2:
        image = np.resize(image, (image.shape[0], image.shape[1], 3))
    image = image[:, :, ::-1]

    # Substract mean pixel values of pascal training set
    if mean is not None:
        image -= mean

    # Reorder multi-channel image matrix from W x H x C to C x H x W expected
    # by Caffe
    image = image.transpose((2, 0, 1))

    return image

    
    
if __name__ == '__main__':

    # Get all options
    args = get_arguments()

    # Run weight printing script
    print "Opening ", args.weights

    # GPU / CPU mode
    if args.cpu:
        print 'Set CPU mode'
        # caffe.set_mode_cpu()
    else:
        print 'Set GPU mode'
        caffe.set_device(1)
        caffe.set_mode_gpu()
    
    # Set the network according to the arguments
    net = caffe.Net(args.model,  # defines the structure of the model
                    args.weights,  # contains the trained weights
                    caffe.TEST)  # use test mode (e.g., don't perform dropout)


    image = preProcessing(np.array(Image.open(args.input)), mean)
    if image is not None:
        net.blobs[dataLayer].reshape(1, *image.shape)
        net.blobs[dataLayer].data[...] = image
    
    net.forward()


    # Which layer to see ?
    layers = args.layers
    if layers is None or len(layers) is 0:
        # display all layers!  Careful with this!
        layers = net.params.keys()
        
    for layer_to_see in layers:

        # Extract weights
        output = net.blobs[layer_to_see].data
        print "OUTPUT SHAPE", output.shape

        # Plot all weights in a window
        output_showable = output.squeeze()[args.filter_num]
        print "OUTPUT SHOWABLE", output_showable.shape, np.min(output_showable), np.max(output_showable)

        cv2.imshow("Output data", (output_showable+args.shift)/args.scale)
        #cv2.imshow("Weights", )
        
        if(args.save != ''):
            cv2.imwrite(args.save+'visualisation_'+str(args.weights)+'.png',
                        data_to_show)

        key = None
        if args.nokey:
            # 2 s between each file iteration
            key = cv2.waitKey(2000)
        else:
            # Stops algorithm and prints weights
            key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key % 256 == 27:  # exit on ESC
            break
