#!/usr/bin/env python

# Code, as generic as possible, for the visualization
# Ex : python /home/pierre/hgRepos/caffeTools/runSegmentation.py --model /home/shared/caffeSegNet/models/segnet_webcam/deploy.prototxt --weights /home/shared/caffeSegNet/models/segnet_webcam/segnet_webcam.caffemodel --colours /home/shared/datasets/CamVid/colours/camvid12.png --output argmax --labels /home/shared/datasets/CamVid/train.txt

# TODO ; check if the video mode work
# TODO ; add the --folder option, that takes a folder as input and segment each image in it
# TODO ; n_cl can be get with net.blobs[layer].channels



import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image
import argparse
import time
import cv2
import os
import sys
from operator import truediv
from utils import score as sc
from utils import iterators
from datetime import datetime

# Caffe module need to be on python path
import caffe

# for some reason, cv2 doesn't define this flag explicitly
CV2_LOAD_IMAGE_UNCHANGED = -1
# ignore divisions by zero
np.seterr(divide='ignore', invalid='ignore')

    

def get_arguments():
    # Import arguments
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    
    # Mandatory options
    parser.add_argument('--model', type=str, required=True, \
                                    help=   'Path to the model (usually [...]/deploy.prototxt)')
    parser.add_argument('--weights', type=str, required=True, \
                                    help=   'Path to the weights (usually [...]/xx.caffemodel)')
    parser.add_argument('--colours', type=str, required=True, \
                                    help=   'If the colours of the classes are provided \
                                            (data/CamVid/colours/camvid12.png)')
    group.add_argument('--video', type=str, \
                                    help=   'A video file to be segmented')
    group.add_argument('--images', type=str, \
                                    help=   'A text file containing a list of images to segment')
    group.add_argument('--labels', type=str, \
                                    help=   'A text file containing space-separated pairs - the first \
                                            item is an image to segment, the second is the ground-truth \
                                            labelling.')
    
    
    # Optional options
    parser.add_argument('--cpu', action="store_true", \
                                    help=   'Default false, set it for CPU mode')
    parser.add_argument('--input', type=str, required=False, default='data', \
                                    help=   'Default is data, change it for the name of the input \
                                            in your network (mostly foundable as the bottom of the \
                                            first layer in the model (prototxt))')
    parser.add_argument('--output', type=str, required=False, default='output', \
                                    help=   'Default is output, change it for the name of the output \
                                            in your network (mostly foundable as the top of the last \
                                            layer in the model (prototxt))')
    parser.add_argument('--key', action="store_true", \
                                    help=   'For visualization image per image (have to press the \
                                            space button for the next image then)')
    parser.add_argument('--PASCAL', action="store_true", \
                                    help=   'If Pascal VOC is used, provide this flag for the mean \
                                            subtraction.')
    parser.add_argument('--hide', action="store_true", \
                                    help=   'If set, won\'t display the results')
    parser.add_argument('--record', type=str, required=False, default='', \
                                    help=   'For recording the videos, expected the path (and prefix) \
                                            of where to save them like /path/to/segnet_. Will create \
                                            two or three videos, depends on if the labels are provided')
    parser.add_argument('--old_caffe', action="store_true", \
                                    help=   'If we use an old version of Caffe (ex. the ones used by \
                                            CRF or DeepLab, the command to create a network in the \
                                            C++ is slightly different.')
    parser.add_argument('--resize', action="store_true", \
                                    help=   'If we want to resize all pictures to the size defined \
                                            by prototxt.')
    
    return parser.parse_args()
    

def build_network(args):
    
    #If using an older version of Caffe, there is no caffe.set_mode_xxx() and we cannot specify caffe.TRAIN or caffe.TEST
    if args.old_caffe: 
            # Creation of the network
            net = caffe.Net(args.model,      # defines the structure of the model
                            args.weights)    # contains the trained weights
    
    else:
            # GPU / CPU mode
            if args.cpu:
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


def pre_processing(img, shape, resize_img):
    # Ensure that the image has the good size
    if resize_img:
        img = img.resize((shape[3], shape[2]), Image.ANTIALIAS)
    
    # Get pixel values and convert them from RGB to BGR
    frame = np.array(img, dtype=np.float32)
    frame = frame[:,:,::-1]

    # Substract mean pixel values of pascal training set
    if args.PASCAL:
        frame -= np.array((104.00698793, 116.66876762, 122.67891434))

    # Reorder multi-channel image matrix from W x H x C to C x H x W expected by Caffe
    frame = frame.transpose((2,0,1))
    
    return frame


def colourSegment(labels, label_colours, input_shape, resize_img):
    # Resize it for 3 channels, now (3, 360, 480)
    if resize_img:
        segmentation_ind_3ch = np.resize(labels, (3, input_shape[2], input_shape[3]))
    else:
        segmentation_ind_3ch = np.resize(labels, (3, labels.shape[0], labels.shape[1]))
    
    # Converts it to format H x W x C (was C x H x W)
    segmentation_ind_3ch = segmentation_ind_3ch.transpose(1,2,0).astype(np.uint8)
    
    # Create a new array (all zeros) with the shape of segmentation_ind_3ch
    _output = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)

    # Fill it with colours of classes
    cv2.LUT(segmentation_ind_3ch, label_colours, _output)

    return _output
        


if __name__ == '__main__':

    # Get all options
    args = get_arguments()
    
    # Set the network according to the arguments
    net = build_network(args)
    
    # Get interesting blobs from the network
    input_blob = net.blobs[args.input]
    output_blob = net.blobs[args.output]
    input_shape = input_blob.data.shape
    
    # Histogram for evan's metrics
    numb_cla = output_blob.channels
    hist = np.zeros((numb_cla, numb_cla))
    
    # Variable for time values
    times = []
    
    # Video recording
    if args.record != '':
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        shape = (input_shape[3],input_shape[2])
        images = cv2.VideoWriter(args.record + 'img.avi', fourcc, 5.0, shape)
        labels = cv2.VideoWriter(args.record + 'labels.avi', fourcc, 5.0, shape)
        segmentation = cv2.VideoWriter(args.record + 'segmentation.avi', fourcc, 5.0, shape)
    
    # Initialize windows
    if args.record == '' and args.hide == False:
        cv2.namedWindow("Input")
        cv2.namedWindow("Output")

    # Create the appropriate iterator
    imageIterator = None
    if args.video is not None:
        imageIterator = iterators.VideoIterator(args.video)
    elif args.images is not None:
        imageIterator = iterators.FileListIterator(args.images)
    elif args.labels is not None:
        imageIterator = iterators.FileListIterator(args.labels, pairs = True)
    else:
        raise Error("No data provided!  Must specify exactly one of --video, --images, or --labels")
    
    # Counter for number of images
    n_im = 0
    
    # Main loop, for each image to process
    for _input, real_label in imageIterator:
        
        # Get the current time
        start = time.time()
        
        # New image
        n_im += 1
        if args.record != '' or args.hide:
            print 'img ' + str(n_im)
        
        # Preprocess the image for the network
        frame = pre_processing(_input, input_shape, args.resize)

        # Shape for input (data blob is N x C x H x W), set data
        input_blob.reshape(1, *frame.shape)
        input_blob.data[...] = frame
        
        # Run the network and take argmax for prediction
        net.forward()
        
        # guessed_label will hold the output of the network and _output the output to display
        _output = 0
        guessed_labels = 0
        
        # Get the output of the network
        guessed_labels = np.squeeze(output_blob.data)
        if len(guessed_labels.shape) == 3:
            guessed_labels = guessed_labels.argmax(axis=0)
        elif len(guessed_labels.shape) != 2:
            print 'Unknown output shape'
            break
        
        # Get the time after the network process
        times.append(time.time() - start)
        
        # Read the colours of the classes
        label_colours = cv2.imread(args.colours).astype(np.uint8)
        
        # Resize input to the same size as other
        if args.resize:
            _input = _input.resize((input_shape[3], input_shape[2]), Image.ANTIALIAS)
        
        # Transform the class labels into a segmented image
        _output = colourSegment(guessed_labels, label_colours, input_shape, args.resize)
        
        # If we also have the ground truth
        if real_label is not None:
            
            # Resize to the same size as other images
            if args.resize:
                real_label = real_label.resize((input_shape[3], input_shape[2]), Image.NEAREST)
            
            # If pascal VOC, reshape the label to HxWx1s
            tmpReal = np.array(real_label)
            if len(tmpReal.shape) == 3:
                tmpReal = tmpReal[:,:,0]
            real_label = Image.fromarray(tmpReal)
            
            # Calculate the metrics for this image
            hist += sc.fast_hist( np.array(real_label).flatten(),
                                  np.array(guessed_labels).flatten(),
                                  numb_cla)
            
            # Convert the ground truth if needed into a RGB array
            show_label = np.array(real_label)
            if len(show_label.shape) == 2:
                show_label = colourSegment(show_label, label_colours, input_shape, args.resize)
            elif len(show_label.shape) != 3:
                print 'Unknown labels format'
            
            # Display the ground truth
            if args.record == '' or args.hide:
                cv2.imshow("Labelled", show_label)
            elif args.record != '':
                labels.write(show_label)
        
        
        # Switch to RGB (PIL Image read files as BGR)
        _input = np.array(cv2.cvtColor(np.array(_input), cv2.COLOR_BGR2RGB))
        
        # Display input and output
        if args.record == '' and args.hide == False:
            cv2.imshow("Input", _input)
            cv2.imshow("Output", _output)
        elif args.record != '':
            images.write(_input)
            segmentation.write(_output)

        # If key, wait for key press, if not, display one image per second
        if args.record == '' and args.hide == False:
            key = 0
            if args.key:
                key = cv2.waitKey(0)
            else:
                key = cv2.waitKey(1000)
            if key % 256 == 27: # exit on ESC - keycode is platform-dependent
                break
    
    
    # Exit properly
    if args.record != '':
        images.release()
        segmentation.release()
        labels.release()
    elif args.hide == False:
        cv2.destroyAllWindows()
    
    # Time elapsed
    print "Average time elapsed when processing one image :\t", sum(times) / float(len(times))
    
    # Display the metrics
    if args.labels is not None:
        
        acc = np.diag(hist).sum() / hist.sum()
        print "Average pixel accuracy :\t", acc
        
        acc = np.diag(hist) / hist.sum(1)
        print "Average mean accuracy :\t\t", np.nanmean(acc)
        
        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        print "Average mean IU score :\t\t", np.nanmean(iu)
        
        freq = hist.sum(1) / hist.sum()
        print "Average frequency weighted IU :\t", (freq[freq > 0] * iu[freq > 0]).sum()
        
        
        print
        print "Mean IU per classes : "
        for idx, i in enumerate(iu):
            print "\tclass ", idx, " : ", i



