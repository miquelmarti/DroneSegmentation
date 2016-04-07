#!/usr/bin/env python

# Code, as generic as possible, for the visualization
# Ex : python /home/pierre/hgRepos/caffeTools/runSegmentation.py --model /home/shared/caffeSegNet/models/segnet_webcam/deploy.prototxt --weights /home/shared/caffeSegNet/models/segnet_webcam/segnet_webcam.caffemodel --colours /home/shared/datasets/CamVid/colours/camvid12.png --output argmax --labels /home/shared/datasets/CamVid/train.txt


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

# Caffe module need to be on python path
import caffe

# for some reason, cv2 doesn't define this flag explicitly
CV2_LOAD_IMAGE_UNCHANGED = -1
# ignore divisions by zero
np.seterr(divide='ignore', invalid='ignore')


class Metric(object):
    def __init__(self, pixel_accuracy, mean_accuracy, mean_IU, freq_weighted_IU, mean_IU_per_class,
                        n_cl, n_im, t_is,
                        n_iis, n_ijs,
                        time_prep, time_arch, time_comp):
        self.pixel_accuracy=pixel_accuracy      # success rate (Pixel accuracy (FCN), or Global average (SegNet))
        self.mean_accuracy=mean_accuracy        # mean accuracy (Class average in SegNet (?))
        self.mean_IU=mean_IU                    # mean IU
        self.freq_weighted_IU=freq_weighted_IU  # frequency weighted IU (for FCN)
        self.mean_IU_per_class=mean_IU_per_class# Mean IU for each class
        self.n_cl=n_cl                          # Number of classes
        self.n_im=n_im                          # Number of images
        self.t_is=t_is                          # Total number of pixels in class i
        # n_ij : The number of pixels of class i predicted to belong to class j):
        self.n_iis=n_iis                        # True positives
        self.n_ijs=n_ijs                        # False positives (wrongly classified)
        self.time_prep=time_prep                # Time to prepare the image to the network
        self.time_arch=time_arch                # Time to process the image with the architecture
        self.time_comp=time_comp                # Time to get the main metrics (like mean IU)

    def __str__(self):
        return  "Metrics :\n" + \
                "\tnb classes : " + str(self.n_cl) + ", nb images : " + str(self.n_im) + "\n" + \
                "\tt_i shape : " + str(np.array(self.t_is).shape) + "\n" + \
                "\t\t" + str(self.t_is) + "\n" + \
                "\tn_iis shape : " + str(np.array(self.n_iis).shape) + "\n" + \
                "\t\t" + str(self.n_iis) + "\n" + \
                "\tn_ijs shape : " + str(np.array(self.n_ijs).shape) + "\n" + \
                "\t\t" + str(self.n_ijs) + "\n"


class FileListIterator(object):
    listFile = None

    def __init__(self, listFileName, pairs=False, sep=' '):
        self.listFile = open(listFileName, 'r')
        self.pairs = pairs
        self.sep = sep
    
    def __iter__(self):
        return self

    def next(self):
        nextLine = self.listFile.next()
        p = nextLine.partition(self.sep)
        nextImg = Image.open(p[0].strip())
        nextLabelImg = None
        if self.pairs:
            nextLabelImg = Image.open(p[2].strip())#cv2.imread(p[2].strip(), CV2_LOAD_IMAGE_UNCHANGED)
        return (nextImg, nextLabelImg)

    def __del__(self):
        if type(self.listFile) is file:
            self.listFile.close()


class VideoIterator(object):
    videoCapture = None

    def __init__(self, videoFileName):
        self.videoCapture = cv2.VideoCapture(videoFileName)
    
    def __iter__(self):
        return self

    def next(self):
        rval, frame = self.videoCapture.read()
        if rval:
            return (Image.fromarray(frame, 'RGB'), None) # no labels for videos
        else:
            raise StopIteration()

    def __del__(self):
        if type(self.videoCapture) is not None:
            self.videoCapture.release()
    

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
    
    return parser.parse_args()
    

def build_network(args):
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


def pre_processing(img, shape):
    # Ensure that the image has the good size
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


def colourSegment(labels, label_colours, input_shape):
    # Resize it for 3 channels, now (3, 360, 480)
    segmentation_ind_3ch = np.resize(labels, (3, input_shape[2], input_shape[3]))
    
    # Converts it to format H x W x C (was C x H x W)
    segmentation_ind_3ch = segmentation_ind_3ch.transpose(1,2,0).astype(np.uint8)
    
    # Create a new array (all zeros) with the shape of segmentation_ind_3ch
    _output = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)

    # Fill it with colours of classes
    cv2.LUT(segmentation_ind_3ch, label_colours, _output)

    return _output


def update_metrics(guess, real, metrics):
    # Delete the borders
    real[real==255]=0
    
    # Goes to look for classes that are present in the groundtruth or guessed segmentation
    class_ranges = range(min(guess.min(), real.min()), max(guess.max(), real.max())+1)
    
    # If we found new classes, we extend the arrays
    if metrics.n_cl < class_ranges[len(class_ranges)-1] + 1:
        wantedNbClasses = class_ranges[len(class_ranges)-1] + 1
        tmpNIJ = np.zeros((len(metrics.n_ijs), wantedNbClasses, wantedNbClasses))
        for im in range(0, metrics.n_im-1):
            for c in range(0, class_ranges[len(class_ranges)-1] + 1 - metrics.n_cl):
                metrics.t_is[im].append(0)
                metrics.n_iis[im].append(0)
            
            for c in range(0, metrics.n_cl):
                for cc in range(0, metrics.n_cl):
                    tmpNIJ[im][c][cc] = metrics.n_ijs[im][c][cc]
            
        metrics.n_ijs = tmpNIJ.tolist()
        metrics.n_cl = class_ranges[len(class_ranges)-1] + 1
    
    # Extend for the current image
    metrics.t_is.append([0]*metrics.n_cl)
    metrics.n_iis.append([0]*metrics.n_cl)
    metrics.n_ijs.append([[0]*metrics.n_cl]*metrics.n_cl)
    
    # For each class
    for ccc in class_ranges:
        tmpNIJ = []
        for ccc2 in range(0, metrics.n_cl):
            # Check the matching between guess and real for the corresponding class
            nij = ((real == ccc) & (guess == ccc2))
            tmpNIJ.append(sum(sum(nij)))
            if ccc == ccc2:
                metrics.n_iis[metrics.n_im-1][ccc] = sum(sum(nij))
        metrics.n_ijs[metrics.n_im-1][ccc] = tmpNIJ
        metrics.t_is[metrics.n_im-1][ccc] = sum(sum((real == ccc)))


def compute_metrics(metrics):
    # For all the images
    for img in range(0, metrics.n_im):
        t_i = np.array(metrics.t_is[img])
        n_ii = np.array(metrics.n_iis[img])
        n_ij = np.array(metrics.n_ijs[img])
        n_cl = len(metrics.t_is[img]) - metrics.t_is[img].count(0)
        
        #print metrics
        
        # pixel_accuracy = sum_i (n_ii) / sum_i (t_i) 
        metrics.pixel_accuracy.append( \
                                np.divide(float(np.sum(n_ii)), float(np.sum(t_i))) )
        
        # mean_accuracy = (1/n_cl) * sum_i (n_ii / t_i) 
        nii_div_ti = np.divide(n_ii, t_i, dtype=float)
        nii_div_ti[np.isnan(nii_div_ti)] = 0
        metrics.mean_accuracy.append( \
                                (1 / float(n_cl)) * np.sum(nii_div_ti) )
        
        # mean_iu = (1/n_cl) * sum_i (n_ii / (t_i + sum_j(n_ji) - n_ii)) 
        ti_plu_nji_min_nii = np.subtract(np.add(t_i, np.sum(n_ij, axis=0)), n_ii)# float(t_i + sum(n_ji) - n_ii)
        nii_div_denom = np.divide(n_ii, ti_plu_nji_min_nii, dtype=float)
        nii_div_denom[np.isnan(nii_div_denom)] = 0
        metrics.mean_IU.append( \
                                (1 / float(n_cl)) * np.sum(nii_div_denom) )
        
        # mean_ius_per_class[i] = n_ii / (t_i + sum_j(n_ji) - n_ii)) 
        metrics.mean_IU_per_class.append(nii_div_denom)
        
        # frequency_weighted_iu = (sum_i (t_i))^-1 * (sum_i ((t_i * n_ii) / (t_i + sum_j(n_ji) - n_ii))) 
        ti_by_nii_div_denom = np.divide(np.multiply(t_i, n_ii), ti_plu_nji_min_nii, dtype=float)
        ti_by_nii_div_denom[np.isnan(ti_by_nii_div_denom)] = 0
        metrics.freq_weighted_IU.append( \
                                (1 / float(sum(t_i))) * np.sum(ti_by_nii_div_denom) )
        


if __name__ == '__main__':

    # Get all options
    args = get_arguments()
    
    # Set the network according to the arguments
    net = build_network(args)
    
    # Get input and output from the network
    input_blob = net.blobs[args.input]
    output_blob = net.blobs[args.output]
    input_shape = input_blob.data.shape
    
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
    
    # Init the structure for storing the metrics
    metrics = Metric([], [], [], [], [], 0, 0, [], [], [], [], [], [])

    # Create the appropriate iterator
    imageIterator = None
    if args.video is not None:
        imageIterator = VideoIterator(args.video)
    elif args.images is not None:
        imageIterator = FileListIterator(args.images)
    elif args.labels is not None:
        imageIterator = FileListIterator(args.labels, pairs = True)
    else:
        raise Error("No data provided!  Must specify exactly one of --video, --images, or --labels")
    
    
    # Main loop, for each image to process
    for _input, real_label in imageIterator:
        
        # New image
        metrics.n_im += 1
        if args.record != '' or args.hide:
            print 'img ' + str(metrics.n_im)
        
        # Preprocess the image for the network
        start = time.time()
        frame = pre_processing(_input, input_shape)
        metrics.time_prep.append(time.time() - start)

        # Shape for input (data blob is N x C x H x W), set data
        input_blob.reshape(1, *frame.shape)
        input_blob.data[...] = frame
        
        # Run the network and take argmax for prediction
        start = time.time()
        net.forward()
        metrics.time_arch.append(time.time() - start)
        
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
        
        # Read the colours of the classes
        label_colours = cv2.imread(args.colours).astype(np.uint8)
        
        # Resize input to the same size as other
        _input = _input.resize((input_shape[3], input_shape[2]), Image.ANTIALIAS)
        
        # Transform the class labels into a segmented image
        _output = colourSegment(guessed_labels, label_colours, input_shape)
        
        # If we also have the ground truth
        if real_label is not None:
            
            # Resize to the same size as other images
            real_label = real_label.resize((input_shape[3], input_shape[2]), Image.ANTIALIAS)
            
            # Calculate the metrics for this image
            start = time.time()
            update_metrics( np.array(guessed_labels, dtype=np.uint8),
                            np.array(real_label, dtype=np.uint8),
                            metrics)
            metrics.time_comp.append(time.time() - start)
            
            # Convert the ground truth if needed into a RGB array
            show_label = np.array(real_label)
            if len(show_label.shape) == 2:
                show_label = colourSegment(show_label, label_colours, input_shape)
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
        else:
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
    
    # Display the metrics
    print "Average time to prepare the image : ", np.mean(metrics.time_prep)
    print "Average time to process the image in the architecture : ", np.mean(metrics.time_arch)
    if len(metrics.t_is) > 0:
        print "Average time to calculate the metrics : ", np.mean(metrics.time_comp)
        compute_metrics(metrics)
        print "Average pixel accuracy : ", np.mean(metrics.pixel_accuracy)
        print "Average mean accuracy : ", np.mean(metrics.mean_accuracy)
        print "Average mean IU score : ", np.mean(metrics.mean_IU)
        print "Average frequency weighted IU : ", np.mean(metrics.freq_weighted_IU)
        print "Mean IU per classes : "
        for i in range(0, metrics.n_cl):
            col = np.array(metrics.mean_IU_per_class)[:,i]
            if np.sum(col != 0) == 0:
                print "\tclass ", i, " : 0"
            else:
                print "\tclass ", i, " : ", (np.sum(col) / np.sum(col != 0))



