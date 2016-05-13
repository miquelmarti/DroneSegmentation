#!/usr/bin/env python

# Code, as generic as possible, for the visualization
# Ex : python /home/pierre/hgRepos/caffeTools/runSegmentation.py --model /home/shared/caffeSegNet/models/segnet_webcam/deploy.prototxt --weights /home/shared/caffeSegNet/models/segnet_webcam/segnet_webcam.caffemodel --colours /home/shared/datasets/CamVid/colours/camvid12.png --output argmax --labels /home/shared/datasets/CamVid/train.txt

# TO DO :
# - check if video mode works
# - add the --folder option, that takes a folder as input and segment each image in it

# - input and output blobs could be looked for automatically in deploy.prototxt instead of using default values in .proto file

import numpy as np
from PIL import Image
import argparse
import time
import cv2
from utils import iterators
import google.protobuf
from scipy import stats

# Caffe module need to be on python path
import caffe

# for some reason, cv2 doesn't define this flag explicitly
CV2_LOAD_IMAGE_UNCHANGED = -1
# ignore divisions by zero
np.seterr(divide='ignore', invalid='ignore')

import ensemble_pb2


# copied from shelhamer's score.py
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k],
                       minlength=n**2).reshape(n, n)


def get_arguments():
    # Import arguments
    parser = argparse.ArgumentParser()
    # Mandatory arguments
    parser.add_argument('ensemble_file', type=str, \
                                    help=   'Link to ensemble.prototxt')
    
    # Optional arguments
    parser.add_argument('--cpu', action="store_true", help='\
    Default false, set it for CPU mode')
    parser.add_argument('--gpu_device', type=int, default=0, help='\
    Default 0, set the GPU device to use')
    uiGroup = parser.add_mutually_exclusive_group()
    uiGroup.add_argument('--key', action="store_true", help='\
    Wait for user to press spacebar after displaying each segmentation.')
    uiGroup.add_argument('--hide', action="store_true", help='\
    Do not display the segmentations, just compute statistics on the dataset.')
    parser.add_argument('--record', type=str, default='', help='\
    For recording the videos, expects the path and file prefix of where to \
    save them (eg. "/path/to/segnet_"). Will create three videos if ground \
    truth is present, three videos if not.')
    
    return parser.parse_args()



def build_network(deploy, weights):
    
    print "Opening Network ", str(weights)
    # Creation of the network
    net = caffe.Net(str(deploy),      # defines the structure of the model
                    str(weights),    # contains the trained weights
                    caffe.TEST)      # use test mode (e.g., don't perform dropout)
    
    return net


def pre_processing(img, shape, resize_img, mean):
    # Ensure that the image has the good size
    if resize_img:
        img = img.resize((shape[3], shape[2]), Image.ANTIALIAS)
    
    # Get pixel values and convert them from RGB to BGR
    frame = np.array(img, dtype=np.float32)
    frame = frame[:,:,::-1]

    # Substract mean pixel values of pascal training set
    #if args.PASCAL:
        #frame -= np.array((104.00698793, 116.66876762, 122.67891434))
    
    frame -= np.array((mean.r, mean.g, mean.b))

    # Reorder multi-channel image matrix from W x H x C to C x H x W expected
    # by Caffe
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


def combineEnsemble(net_outputs, method):
    output = 0
    
    #If there is only one model, we skip this step
    if np.asarray(net_outputs).shape[0] == 1:
        return np.squeeze(net_outputs)
    
    if method==1: #Majority voting
            #Calculates the label (by looking at the maximum class score)
            net_outputs = np.squeeze(np.asarray(net_outputs).argmax(axis=1))
            
            #Looks for the most common label for each pixel
            output = np.squeeze(stats.mode(net_outputs, axis=0)[0]).astype(int)
            print output.shape
    
    if method==2: #Logit averaging
            output = np.zeros(net_outputs[0].shape)
            
            for current_net_output in net_outputs:
                output = output + current_net_output
    
    if method==3: #Probability averaging
            output = np.zeros(net_outputs[0].shape)
            
            for current_net_output in net_outputs:
                output = output + softmax(current_net_output)
    
    
    return output



if __name__ == '__main__':

    # Get all options
    args = get_arguments()
    
    # GPU / CPU mode
    if args.cpu:
        print 'Set CPU mode'
        caffe.set_mode_cpu()
    else:
        print 'Set GPU mode'
        caffe.set_device(args.gpu_device)
        caffe.set_mode_gpu()
    
    #Create an Ensemble object
    ensemble = ensemble_pb2.Ensemble()

    # Read the ensemble.prototxt
    f = open(args.ensemble_file, "rb")
    google.protobuf.text_format.Merge(f.read(), ensemble)
    f.close()
    
    if not ensemble.IsInitialized():
        raise ValueError('Prototxt not complete (not all required fields are present)')
    
    
    #Loading all neural networks
    input_blob = []
    output_blob = []
    nets = []
    nb_net = 0 #Counter of nets
    for model in ensemble.model:
        #Create network and add it to network list
        nets.append(build_network(model.deploy, model.weights))
        
        #Get information about blobs
        input_blob.append(nets[nb_net].blobs[model.input])
        output_blob.append(nets[nb_net].blobs[model.output])
        nb_net = nb_net+1
            
    input_shape = input_blob[0].data.shape
    
    # Histogram for evan's metrics
    numb_cla = output_blob[0].channels
    hist = np.zeros((numb_cla, numb_cla))
    
    times = [] #Variable for test times
    
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
    if ensemble.input.type == 1:
        imageIterator = iterators.VideoIterator(ensemble.input.file)
    elif ensemble.input.type == 2:
        imageIterator = iterators.FileListIterator(ensemble.input.file)
    elif ensemble.input.type == 3:
        imageIterator = iterators.FileListIterator(ensemble.input.file, pairs = True)
    else:
        raise Error("No data provided in the prototxt!")
    
    n_im = 0 #Image counter
    
    # Main loop, for each image to process
    for _input, real_label in imageIterator:
        
        # Get the current time
        start = time.time()
        
        # Preprocess the image for the network
        frame = pre_processing(_input, input_shape, ensemble.input.resize, ensemble.input.mean)

        
        guessed_labels = []
        nb_net = 0
        for net in nets:
                
                # Shape for input (data blob is N x C x H x W), set data
                input_blob[nb_net].reshape(1, *frame.shape)
                input_blob[nb_net].data[...] = frame
                
                # Run the network
                net.forward()
                
                #Get output of the network
                guessed_labels.append(np.squeeze(output_blob[nb_net].data))
                nb_net = nb_net+1
        
        #Combine the outputs of each net by the chosen method (voting, averaging, etc.)
        guessed_labels = combineEnsemble(guessed_labels, ensemble.ensemble_type)
        
        # guessed_label will hold the output of the network and _output the output to display
        _output = 0
        
        # Get the output of the network
        if len(guessed_labels.shape) == 3:
            guessed_labels = guessed_labels.argmax(axis=0)
        elif len(guessed_labels.shape) != 2:
            print 'Unknown output shape'
            break
        
        # Get the time after the network process
        times.append(time.time() - start)
        
        # Read the colours of the classes
        label_colours = cv2.imread(ensemble.input.colours).astype(np.uint8)
        
        # Resize input to the same size as other
        if ensemble.input.resize:
            _input = _input.resize((input_shape[3], input_shape[2]), Image.ANTIALIAS)
        
        # Transform the class labels into a segmented image
        _output = colourSegment(guessed_labels, label_colours, input_shape, ensemble.input.resize)
        
        # If we also have the ground truth
        if real_label is not None:
            
            # Resize to the same size as other images
            if ensemble.input.resize:
                real_label = real_label.resize((input_shape[3], input_shape[2]), Image.NEAREST)
            
            # If pascal VOC, reshape the label to HxWx1s
            tmpReal = np.array(real_label)
            if len(tmpReal.shape) == 3:
                tmpReal = tmpReal[:,:,0]
            real_label = Image.fromarray(tmpReal)
            
            # Calculate the metrics for this image
            difference_matrix = fast_hist( np.array(real_label).flatten(),
                                  np.array(guessed_labels).flatten(),
                                  numb_cla)
            hist += difference_matrix
            
            # Print image number and accuracy
            if args.record != '' or args.hide:
                n_im += 1
                print 'Image ', str(n_im), " accuracy: ", float(np.diag(difference_matrix).sum()) / difference_matrix.sum()
            
            # Convert the ground truth if needed into a RGB array
            show_label = np.array(real_label)
            if len(show_label.shape) == 2:
                show_label = colourSegment(show_label, label_colours, input_shape, ensemble.input.resize)
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
    if ensemble.input.type == 3: #If labels exist
        
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



