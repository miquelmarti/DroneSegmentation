#!/usr/bin/env python

# Code, as generic as possible, for the visualization

# TO DO :
# - check if video mode works
# - add the --folder option, that takes a folder as input and segment each image in it

# - input and output blobs could be looked for automatically in deploy.prototxt instead of using default values in .proto file


# Caffe module need to be on python path
import caffe
import numpy as np
from PIL import Image
import argparse
import time
import cv2
import google.protobuf
from scipy import stats
import ensemble_pb2
from caffeUtils import iterators, score, protoUtils
import os
from os.path import basename


# for some reason, cv2 doesn't define this flag explicitly
CV2_LOAD_IMAGE_UNCHANGED = -1
# ignore divisions by zero
np.seterr(divide='ignore', invalid='ignore')

#If no networks provide input shape, this will be the default one (there could be a nicer way of doing this, by loading .npy arrays and have a direct look at their size)
default_input_shape = np.array((1,21,500,500))
default_numb_cla = 21


# copied from shelhamer's score.py
# def fast_hist(a, b, n):
#     k = (a >= 0) & (a < n)
#     return np.bincount(n * a[k].astype(int) + b[k],
#                        minlength=n**2).reshape(n, n)


def get_arguments():
    # Import arguments
    parser = argparse.ArgumentParser()
    # Mandatory arguments
    parser.add_argument('ensemble_file', type=str, help='\
    Link to config.prototxt')
    # Optional arguments
    parser.add_argument('--cpu', action="store_true", help='\
    Default false, set it for CPU mode')
    parser.add_argument('--gpu_device', type=int, default=None, help='\
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
    parser.add_argument('--crop', type=int, default=1, help='\
    Cut each image into crop*crop small images before inputting in network (for GPU memory saving)')
    parser.add_argument('--view_resize', type=float, default=1.0, help='\
    Cut each image into crop*crop small images before inputting in network (for GPU memory saving)')
    parser.add_argument('--show_prob', type=int, help='\
    If provided, will display the probability of the given class at each \
    pixel.') 
    return parser.parse_args()


def build_network(deploy, weights):
    print "Opening Network ", str(weights)
    # Creation of the network
    net = caffe.Net(str(deploy),      # defines the structure of the model
                    str(weights),    # contains the trained weights
                    caffe.TEST)      # use test mode (e.g., don't perform dropout)
    
    return net


# def pre_processing(img, shape, resize_img, mean):
#     # Ensure that the image has the good size
#     if resize_img:
#         img = img.resize((shape[3], shape[2]), Image.ANTIALIAS)
    
#     # Get pixel values and convert them from RGB to BGR
#     frame = np.array(img, dtype=np.float32)
#     frame = frame[:,:,::-1]

#     # Substract mean pixel values of pascal training set
#     #if args.PASCAL:
#         #frame -= np.array((104.00698793, 116.66876762, 122.67891434))
    
#     frame -= np.array((mean.r, mean.g, mean.b))

#     # Reorder multi-channel image matrix from W x H x C to C x H x W expected
#     # by Caffe
#     frame = frame.transpose((2,0,1))
    
#     return frame


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
    

def softmax(x): #Softmax function, transforming logits into probabilities
    out = np.exp(x)
    out = out/np.sum(out, axis=0)
    return out

    
def combineEnsemble(net_outputs, method, weighting):
    output = 0
    
    #If there is only one model, we skip this step
    if np.asarray(net_outputs).shape[0] == 1:
        return np.squeeze(net_outputs)
    
    if method==ensemble_pb2.VOTING: #Majority voting
            #Calculates the label (by looking at the maximum class score)
            net_outputs = np.squeeze(np.asarray(net_outputs).argmax(axis=1))
            
            #Looks for the most common label for each pixel
            output = np.squeeze(stats.mode(net_outputs, axis=0)[0]).astype(int)
    
    if method==ensemble_pb2.LOGITARI: #Logit arithmetic averaging
            output = np.zeros(net_outputs[0].shape)
            
            sum_weightings = 0
            for current_net_output, current_weighting in zip(net_outputs,
                                                             weighting):
                output = output + current_net_output * current_weighting
                sum_weightings = sum_weightings + current_weighting
            
            #Make it a mean instead of a sum of logits
            output = output/sum_weightings
            
    if method==ensemble_pb2.LOGITGEO: #Logit geometric averaging
            output = np.ones(net_outputs[0].shape)
            
            for current_net_output in net_outputs:
                output = np.multiply(output, current_net_output)
                print current_net_output
            
            print output
            
            #Make it a mean
            output = np.power(output,float(1)/len(net_outputs))
    
    if method==ensemble_pb2.PROBAARI: #Probability arithmetic averaging
            output = np.zeros(net_outputs[0].shape)
            
            sum_weightings = 0
            for current_net_output, current_weighting in zip(net_outputs, weighting):
                output = output + softmax(current_net_output) * current_weighting
                sum_weightings = sum_weightings + current_weighting
            
            #Make it a mean instead of a sum of probabilities
            output = output/sum_weightings
    
    if method==ensemble_pb2.PROBAGEO: #Probability geometric averaging
            output = np.ones(net_outputs[0].shape)
            
            for current_net_output in net_outputs:
                output = np.multiply(output, softmax(current_net_output))
            
            #Make it a mean
            output = np.power(output,float(1)/len(net_outputs))
    
    return output

def cropAndSegment(_input, newShape, args, net, in_blob, out_blob, mean): #This function segments the image patch by patch by cropping
        _input_array = np.asarray(_input)
        guessed_label = np.zeros((numb_cla,newShape[2]*args.crop,newShape[3]*args.crop)) #Set output segmentation shape

        for id_h in range(0,args.crop): #Iterate cropping along first dimension
                crop_h = newShape[2] #Height of the cropped image we input to net
                h1 = id_h*crop_h #Current crop position in height
                
                for id_w in range(0,args.crop): #Iterate cropping along second dimension
                        crop_w = newShape[3] #Width of cropped image we input to net
                        w1 = id_w*crop_w #Current crop position in width
                        
                        #Crop image
                        cropped = _input_array[h1:(h1+crop_h), w1:(w1+crop_w), :]
                        
                        #Get segmentation with cropped image
                        guessed_label[:,h1:(h1+crop_h),w1:(w1+crop_w)] = score.segmentImage(net, Image.fromarray(cropped), in_blob, out_blob, mean, newShape=newShape)[0]
        
        return guessed_label

if __name__ == '__main__':

    # Get all options
    args = get_arguments()
    
    # GPU / CPU mode
    if args.cpu:
        print 'Set CPU mode'
        caffe.set_mode_cpu()
    else:
        print 'Set GPU mode'
        if args.gpu_device is not None:
                caffe.set_device(args.gpu_device)
        caffe.set_mode_gpu()
    
    # Read the ensemble.prototxt
    config = ensemble_pb2.Ensemble()
    protoUtils.readFromPrototxt(config, args.ensemble_file)
    
    if not config.IsInitialized():
        raise ValueError('\
        Prototxt not complete (not all required fields are present)')
    
    # Loading all neural networks
    input_blobs = []
    output_blobs = []
    nets = []
    model_weighting = []
    
    for model in config.model: #Load all networks from the different models specified in .proto
        #Create network and add it to network list
        nets.append(build_network(model.deploy, model.weights))
        
        #Get information about input and output layers
        input_blobs.append(model.input)
        output_blobs.append(model.output)
        model_weighting.append(model.weighting)
        
    for model in config.modelOutput: #Load all model output folders specified in .proto
        #Create network and add it to network list
        nets.append(None)
        
        #Get information about input and output layers
        input_blobs.append(None)
        output_blobs.append(model.folder)
        model_weighting.append(model.weighting)
    
    if input_blobs[0] is not None:        
        input_shape = nets[0].blobs[input_blobs[0]].data.shape
        numb_cla = nets[0].blobs[output_blobs[0]].channels
    else:
        input_shape = default_input_shape
        numb_cla = default_numb_cla
    
    # Histogram for evan's metrics
    numb_cla = nets[0].blobs[output_blobs[0]].channels
    totalHist = np.zeros((numb_cla, numb_cla))
    
    times = []  # Variable for test times
    
    # Video recording
    if args.record != '':
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        shape = (input_shape[3], input_shape[2])
        images = cv2.VideoWriter(args.record + 'img.avi', fourcc, 5.0, shape)
        labels = cv2.VideoWriter(args.record + 'labels.avi', fourcc, 5.0,
                                 shape)
        segmentation = cv2.VideoWriter(args.record + 'segmentation.avi',
                                       fourcc, 5.0, shape)
    
    # Initialize windows
    if args.record == '' and not args.hide:
        cv2.namedWindow("Input")
        cv2.namedWindow("Output")

    # Create the appropriate iterator
    imageIterator = None
    if config.input.type == ensemble_pb2.Input.VIDEO:
        imageIterator = iterators.VideoIterator(config.input.file)
    elif config.input.type == ensemble_pb2.Input.IMAGES:
        imageIterator = iterators.FileListIterator(config.input.file)
    elif config.input.type == ensemble_pb2.Input.LABELS:
        imageIterator = iterators.FileListIterator(config.input.file,
                                                   pairs=True)
    else:
        raise ValueError("No data provided in the prototxt!")
        
        
    #If previse an output folder, create a list.txt file for summarising
    # which numpy matrices of which images have been created
    if config.outputFolder != "None":
        summaryFile = open(config.outputFolder+"list.txt", 'w')
    
    # process each image, one-by-one
    n_im = 0  # Image counter
    
    mean = np.array([config.input.mean.r,
                     config.input.mean.g,
                     config.input.mean.b])
                     
    for _input, real_label, imagePath in imageIterator:
        n_im += 1
        guessed_labels = []
        start = time.time()
        
        # Extracts current image name from the image path
        imageName = ''
        if imagePath: #Bug, because sometimes iterators (video namely) return None imagePath
                imageName = basename(os.path.splitext(imagePath)[0])
        
        for net, in_blob, out_blob in zip(nets, input_blobs, output_blobs):
            guessed_label = None
            newShape = None
            
            if net is None: #If no net is provided, it means the outputs are in .npy files
                guessed_label = np.load(out_blob+imageName+'.npy')
                guessed_labels.append(np.squeeze(guessed_label))
                continue
            
            if config.input.resize: #Set new shape to shape defined by .prototxt
                newShape = input_shape
            else:
                newShape = np.array([1,3,int(np.asarray(_input).shape[0]/float(args.crop)),int(np.asarray(_input).shape[1]/float(args.crop))])
            #newShape is the shape of images that will be input to the network, so it already includes the cropping (for example 2000x2000 images, cropped to 500x500 will have newShape=(1,3,500,500)
                
            #If want to crop images before inputting the network (saves memory)
            if args.crop > 1:
                guessed_label = cropAndSegment(_input, newShape, args, net, in_blob, out_blob, mean)
                                
            else: #If no cropping, simply get segmentation with input image
                guessed_label = score.segmentImage(net, _input, in_blob, out_blob,
                                               mean, newShape=newShape)[0]
            guessed_labels.append(np.squeeze(guessed_label)) #Add network output to list

        # Combine the outputs of each net by the chosen method (voting,
        # averaging, etc.)
        guessed_labels = combineEnsemble(guessed_labels,
                                         config.ensemble_type, 
                                         model_weighting)
                                         
        
        #If we precise an output folder in the prototxt
        if config.outputFolder != "None":
                # Saves network outputs to numpy file
                np.save(config.outputFolder+imageName+".npy",guessed_labels)
                # Writes down in list.txt the files that have been saved
                summaryFile.write(imagePath+" "+config.outputFolder+imageName+".npy\n")

        
        # Get the time after the network process
        times.append(time.time() - start)
        
        # Get the output of the network
        # TODO: Not sure of the first if, but was madatory for making resnet works
        if len(guessed_labels.shape) == 1 and len(guessed_labels[0].shape) == 4:
            guessed_labels = guessed_labels[0][0]
        if len(guessed_labels.shape) == 3:
            guessed_labels = guessed_labels.argmax(axis=0)
        elif len(guessed_labels.shape) != 2:
            print 'Unknown output shape:', guessed_labels.shape
            break

        # Read the colours of the classes
        label_colours = cv2.imread(config.input.colours).astype(np.uint8)
        
        # Resize input to the same size as other
        if config.input.resize:
            _input = _input.resize((input_shape[3], input_shape[2]),
                                   Image.ANTIALIAS)
        
        # Transform the class labels into a segmented image
        guessed_image = colourSegment(guessed_labels, label_colours,
                                      input_shape, config.input.resize)
        
        # If we also have the ground truth
        if real_label is not None:
            
            # Resize to the same size as other images
            if config.input.resize:
                real_label = real_label.resize((input_shape[3],
                                                input_shape[2]), Image.NEAREST)
            
            # If pascal VOC, reshape the label to HxWx1s
            tmpReal = np.array(real_label)
            if len(tmpReal.shape) == 3:
                tmpReal = tmpReal[:, :, 0]
            real_label = Image.fromarray(tmpReal)

            # Calculate the histogram for this image
            gtArray = np.array(real_label).flatten()
            guessArray = np.array(guessed_labels).flatten()
            hist = score.fast_hist(gtArray, guessArray, numb_cla)
            totalHist += hist

            # Print image number and accuracy
            if args.record != '' or args.hide:
                acc = score.computeSegmentationScores(hist).overallAcc
                print 'Image', n_im, "(", imageName ,") accuracy:", acc
            
            # Convert the ground truth if needed into a RGB array
            gt_image = np.array(real_label)
            if len(gt_image.shape) == 2:
                gt_image = colourSegment(gt_image, label_colours,
                                         input_shape, config.input.resize)
            elif len(gt_image.shape) != 3:
                print 'Unknown labels format'
            
            # Display the ground truth
            if args.record == '' or args.hide:
                newSize = (int(args.view_resize*gt_image.shape[1]),
                           int(args.view_resize*gt_image.shape[0]))
                labeledImage = cv2.resize(gt_image, newSize)
                cv2.imshow("Ground truth", labeledImage)
            elif args.record != '':
                labels.write(gt_image)
                
        # Switch to RGB (PIL Image read files as BGR)
        if len(np.array(_input).shape) is 3:
            _input = np.array(cv2.cvtColor(np.array(_input),
                                           cv2.COLOR_BGR2RGB))
        
        # Display input and output
        if args.record == '' and not args.hide:
            newSize = (int(args.view_resize*_input.shape[1]),
                       int(args.view_resize*_input.shape[0]))
            cv2.imshow("Input", cv2.resize(_input, newSize))
            if args.show_prob is not None:
                probs = softmax(logits)
                showProb = probs[args.show_prob]
                showProb = showProb / np.max(showProb)
                cv2.imshow("probability", showProb)
            else:
                newSize = (int(args.view_resize*guessed_image.shape[1]),
                           int(args.view_resize*guessed_image.shape[0]))
                cv2.imshow("Output", cv2.resize(guessed_image, newSize))
            key = 0
            if args.key:
                key = cv2.waitKey(0)
            else:
                key = cv2.waitKey(2000)
            if key % 256 == 27:  # exit on ESC - keycode is platform-dependent
                break
        elif args.record != '':
            images.write(_input)
            segmentation.write(guessed_image)

    # Exit properly
    if args.record != '':
        images.release()
        segmentation.release()
        labels.release()
    elif not args.hide:
        cv2.destroyAllWindows()
        
    #If we opened an output file, close it
    if config.outputFolder != "None":
        summaryFile.close()
    
    # Time elapsed
    avgImageTime = sum(times) / float(len(times))
    print "Average time elapsed when processing one image :\t", avgImageTime
    
    if config.input.type == ensemble_pb2.Input.LABELS:
        # Display the metrics
        scores = score.computeSegmentationScores(totalHist)
        print "Average pixel accuracy :\t", scores.overallAcc
        print "Average mean accuracy :\t\t", scores.meanAcc
        print "Average mean IU score :\t\t", scores.meanIu
        print "Average frequency weighted IU :\t", scores.fwavacc

        print
        print "Mean IU per classes : "
        for idx, i in enumerate(scores.iu):
            print "\tclass ", idx, " : ", i
