#!/usr/bin/env python

# Code for visualizing the main look of the weights
# Ex : python view_weights.py --model /home/johannes/project/models/segnet/deploy-pascal.prototxt --weights /home/johannes/project/models/segnet/train_pascal/ --layers "conv1_1" --scale 0.1 --shift -0.05


import numpy as np
import argparse
import cv2
import caffe


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
    
    # Optional options
    parser.add_argument('--cpu', action="store_true", help='\
    Default False, set True for CPU mode')
    parser.add_argument('--terminal', action="store_true", help='\
    Set true for printing weights in terminal')
    parser.add_argument('--shift', type=float, default=0.0, help='\
    Substracts the data value by this before plot')
    parser.add_argument('--scale', type=float, default=1.0, help='\
    Divides the data value by this before plot')
    parser.add_argument('--autoscale', action="store_true", help='\
    Sets automatically the scale to better print the weights')
    parser.add_argument('--nokey', action="store_true", help='\
    Move through the weights quickly')
    parser.add_argument('--save', type=str, default='', help='\
    Saves png files in this folder')
    
    return parser.parse_args()

    
def vis_square(data, shift=0, scale=1, autoscale=False):
    """Code by Caffe.
    Take an array of shape (n, height, width) or (n, height, width, 3)
    and visualize each (height, width) thing in a grid of size approx.
    sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    if data.max()-data.min() > 0:
            
        if autoscale:
            shift = data.min()
            scale = data.max() - data.min()
            # data = (data - data.min()) / (data.max() - data.min())
        # else:
        data = (data - shift) / scale

        # force the number of filters to be square
        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = (((0, n ** 2 - data.shape[0]),
                   (0, 1), (0, 1)) +  # add some space between filters
                   ((0, 0),) * (data.ndim - 3))  # don't pad last dim

        # pad with ones (white)
        data = np.pad(data, padding, mode='constant', constant_values=1)

        # tile the filters into an image
        tempShape = (n, n) + data.shape[1:]
        axesOrder = (0, 2, 1, 3) + tuple(range(4, data.ndim + 1))
        data = data.reshape(tempShape).transpose(axesOrder)
        finalShape = (n * data.shape[1], n * data.shape[3]) + data.shape[4:]
        data = data.reshape(finalShape)

        # Plot a 600x600 window of the weights
        data_to_show = cv2.resize(data, (600, 600), 0, 0, 0,
                                  cv2.INTER_NEAREST)

        return shift, scale, data_to_show
            
    else:
        print "Filter weights are all equal to ", data.max(), " !"
        return data.max()
    
    
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
        caffe.set_mode_gpu()
    
    # Set the network according to the arguments
    net = caffe.Net(args.model,  # defines the structure of the model
                    args.weights,  # contains the trained weights
                    caffe.TEST)  # use test mode (e.g., don't perform dropout)

    # Which layer to see ?
    layers = args.layers
    if layers is None or len(layers) is 0:
        # display all layers!  Careful with this!
        layers = net.params.keys()
        
    for layer_to_see in layers:

        # Extract weights
        weights = net.params[layer_to_see][0].data
        if args.terminal:
            print ''.join(["WEIGHTS of layer", layer_to_see, " (of shape ",
                           weights.shape, ") : ", weights])
        # Weight format :  weights[filter_number_to_show,channel_to_show,:,:]

        # Extract bias
        if args.terminal and len(net.params[layer_to_see] > 1):
            bias = net.params[layer_to_see][1].data
            print "BIAS (of shape ", bias.shape, ") : ", bias

        # Transforms a little the data in order to make them printable
        # Puts dimensions in right order
        weights_printable = weights.transpose(0, 2, 3, 1)

        if weights_printable.shape[3] > 3:
            # In case there are more than 3 channels, show the 3 first ones
            weights_printable = weights_printable[:, :, :, 0:3]

        # Plot all weights in a window
        shift, scale, data_to_show = vis_square(weights_printable, args.shift,
                                                args.scale, args.autoscale)
        print ','.join([layer_to_see, str(shift), str(scale)])
        cv2.imshow(layer_to_see, data_to_show)
        
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
