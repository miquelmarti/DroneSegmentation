import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import argparse
import time
import cv2
import os

# Caffe module need to be on python path
import caffe

# for some reason, cv2 doesn't define this flag explicitly
CV2_LOAD_IMAGE_UNCHANGED = -1 

def get_arguments():
    # Import arguments
    parser = argparse.ArgumentParser()
    
    # Mandatory options
    parser.add_argument('--model', type=str, required=True, help='Path to the model (usually [...]/deploy.prototxt)')
    parser.add_argument('--weights', type=str, required=True, help='Path to the weights (usually [...]/xx.caffemodel)')
    
    # Optional options
    parser.add_argument('--colours', type=str, required=False, default='', help='If the colours of the classes are provided (data/CamVid/colours/camvid12.png)')
    parser.add_argument('--gpu', type=bool, required=False, default=True, help='Default true, set False for CPU mode')
    parser.add_argument('--test', type=bool, required=False, default=True, help='Default true, set False for training mode')
    parser.add_argument('--input', type=str, required=False, default='data', help='Default is data, change it for the name of the input in your network (mostly foundable as the bottom of the first layer in the model (prototxt))')
    parser.add_argument('--output', type=str, required=False, default='output', help='Default is output, change it for the name of the output in your network (mostly foundable as the top of the last layer in the model (prototxt))')

    parser.add_argument('--FCN', type=bool, required=False, default=False, help='If FCN is used, set this argument to true. Because an extra argmax at the output is needed (FCN returns class scores, while SegNet directly returns class index)')
    
    # Mandatory arguments
    parser.add_argument('data_to_segment', type=str, help='a text file containing a list of images to segment')
    parser.add_argument('--labels', action="store_true", help='if provided, it means the files in the data .txt file are paired into data,label pairs.')
    
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
    if args.test == False:
        net = caffe.Net(args.model,      # defines the structure of the model
                        args.weights,    # contains the trained weights
                        caffe.TRAIN)     # use train mode (e.g., don't perform dropout)
    else:
        net = caffe.Net(args.model,      # defines the structure of the model
                        args.weights,    # contains the trained weights
                        caffe.TEST)      # use test mode (e.g., don't perform dropout)
    
    return net


def pre_processing(img, shape):
    # Ensure that the image has the good size
    # img = cv2.resize(img, (shape[3], shape[2])) # doesn't work for Image format
    
    # Get pixel values and convert them from RGB to BGR
    frame = np.array(img, dtype=np.float32)
    frame = frame[:,:,::-1]

    # Substract mean pixel values of VGG training set
    frame -= np.array((104.00698793,116.66876762,122.67891434))

    # Reorder multi-channel image matrix from W x H x C to C x H x W expected by Caffe
    frame = frame.transpose((2,0,1))
    
    return frame


def compute_mean_IU(guess,real):
    class_ranges = range(min(guess.min(),real.min()),max(guess.max(),real.max())+1)
    number_classes = 0
    IU_value = 0.0

    for ccc in class_ranges:
        true_positive  = sum(sum((real == ccc) & (guess == ccc)))
        false_positive = sum(sum((real != ccc) & (guess == ccc)))
        false_negative = sum(sum((real == ccc) & (guess != ccc)))

        if true_positive+false_positive+false_negative>0:
            IU_value = IU_value + true_positive/float(true_positive+false_positive+false_negative)
            number_classes = number_classes+1

    result = 1/number_classes
    result = result*IU_value

    return IU_value/number_classes



def colourSegment(labels, label_colours, input_shape): 
    #colourSegment function transforms the class labels into a viewable image

    # Resize it for 3 channels, now (3, 360, 480)
    segmentation_ind_3ch = np.resize(labels,(3,input_shape[2],input_shape[3]))
    # Converts it to format H x W x C (was C x H x W)
    segmentation_ind_3ch = segmentation_ind_3ch.transpose(1,2,0).astype(np.uint8)
    # Create a new array (all zeros) with the shape of segmentation_ind_3ch
    _output = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)

    # Fill it with colours of classes
    cv2.LUT(segmentation_ind_3ch,label_colours,_output)
    _output = _output.astype(float)/255 # Optional, for convention

    return _output
        


if __name__ == '__main__':

    # Get all options
    args = get_arguments()
    
    # Set the network according to the arguments
    net = build_network(args)
    
    # Get input and output from the network
    input_blob = net.blobs[args.input]
    output_blob = net.blobs[args.output]
    
    
    # Display windows
    cv2.namedWindow("Input")
    cv2.namedWindow("Output")

    mean_IUs = []
    imageListFile = open(args.data_to_segment, 'r')
    for line in imageListFile:
        filename = None
        if args.labels:
            filename = line.partition(' ')[0]
        else:
            filename = line.strip()

        # Given an Image, convert to ndarray and preprocess for VGG
        _input = cv2.imread(filename, CV2_LOAD_IMAGE_UNCHANGED)
        frame = pre_processing(_input, input_blob.data.shape)

        # Shape for input (data blob is N x C x H x W), set data
        input_blob.reshape(1, *frame.shape)
        input_blob.data[...] = frame

        # Run the network and take argmax for prediction
        net.forward()
        _output = 0
        guessed_labels = 0


        # If FCN-8
        if args.FCN:
            # Squeeze : matrix size (1, 1, 360, 480) => (360, 480).
            # Take argmax to choose the biggest class score (FCN outputs class scores)
            guessed_labels = np.argmax(np.squeeze(output_blob.data), axis=0)

        # If SegNet
        else:
            # Squeeze : matrix size (1, 1, 360, 480) => (360, 480).
            guessed_labels = np.squeeze(output_blob.data)

        # Read the colours of the classes
        label_colours = cv2.imread(args.colours).astype(np.uint8)
        input_shape = input_blob.data.shape
        #Transform the class labels into a segmented image
        _output = colourSegment(guessed_labels, label_colours, input_shape)

        #If specify the ground-truth, calculate mean IU
        if args.labels:
            #Load ground-truth
            label_filename = line.partition(' ')[2].strip()
            real_label = np.array(cv2.imread(label_filename,
                                             CV2_LOAD_IMAGE_UNCHANGED))

            #Calculate and print the mean IU
            mean_IU = compute_mean_IU(np.array(guessed_labels, dtype=np.uint8),
                                      np.array(real_label, dtype=np.uint8))
            print 'Mean IU:', mean_IU
            mean_IUs.append(mean_IU)

            #Transform the real labels into a showable image
            show_label = colourSegment(real_label, label_colours, input_shape)
            cv2.imshow("Labelled", show_label)

        # Display input and output
        cv2.imshow("Input", np.array(_input))
        cv2.imshow("Output", _output)

        key = cv2.waitKey(1)
        if key == 27: # exit on ESC
            break

    cv2.destroyAllWindows()
    print "Average mean IU score:", np.mean(mean_IUs)
