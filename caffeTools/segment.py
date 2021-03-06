#!/usr/bin/env python

# Caffe module need to be on python path
import caffe
import numpy as np
from PIL import Image
import argparse
import time
import cv2
from scipy import stats
import ensemble_pb2
from caffeUtils import iterators, score, protoUtils
import os
from os.path import basename


# for some reason, cv2 doesn't define this flag explicitly
CV2_LOAD_IMAGE_UNCHANGED = -1
# ignore divisions by zero
np.seterr(divide='ignore', invalid='ignore')

# If no networks provide input shape, this will be the default one (there could
# be a nicer way of doing this, by loading .npy arrays and have a direct look
# at their size)
default_input_shape = np.array((1, 10, 1920, 1080))
default_numb_cla = 10


class LogitCollection(object):

    def __init__(self, logit_col_buf):
        self.directory = logit_col_buf.folder
        self.weight = logit_col_buf.weighting

    def getImageLogits(self, image_basename):
        image_basename = os.path.splitext(image_basename)[0]
        logit_filename = self.directory + image_basename + '.npy'
        return np.load(logit_filename)


class Model(object):

    def __init__(self, model_buf):
        self.net = build_network(model_buf.deploy, model_buf.weights)
        self.input_layer = model_buf.input
        self.output_layer = model_buf.output
        self.weight = model_buf.weighting

    def segmentImage(self, input_image, mean, new_shape=None):
        # TODO should we pass mean here or in the constructor?
        seg = score.segmentImage(self.net, input_image, self.input_layer,
                                 self.output_layer, mean)
        return seg

    def getInputShape(self):
        return self.net.blobs[self.input_layer].data.shape

    def getNumClasses(self):
        return self.net.blobs[self.output_layer].channels


def get_arguments():
    # Import arguments
    parser = argparse.ArgumentParser()
    # Mandatory arguments
    parser.add_argument('ensemble_file', type=str, help='\
    Link to config.prototxt')
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
    parser.add_argument('--crop', type=int, default=1, help='\
    Cut each image into crop*crop small images before inputting in network \
    (for GPU memory saving)')
    parser.add_argument('--view_resize', type=float, default=1.0, help='\
    Display images downscaled by this amount')
    parser.add_argument('--show_prob', type=int, help='\
    If provided, will display the probability of the given class at each \
    pixel.')
    parser.add_argument('--center', type=float, default=1.0, help='\
    Crops the central part of the image, with a sizeequal to args.center \
    times the original size')
    return parser.parse_args()


def build_network(deploy, weights):
    print "Opening Network ", str(weights)
    # Creation of the network
    net = caffe.Net(str(deploy),      # defines the structure of the model
                    str(weights),    # contains the trained weights
                    caffe.TEST)  # use test mode (e.g., don't perform dropout)

    return net


def colourSegment(labels, label_colours):
    # Resize it for 3 channels, now (3, 360, 480)
    segmentation_ind_3ch = np.resize(labels, (3,) + labels.shape)

    # Converts it to format H x W x C (was C x H x W)
    segmentation_ind_3ch = segmentation_ind_3ch.transpose(1, 2, 0)
    segmentation_ind_3ch = segmentation_ind_3ch.astype(np.uint8)

    # Colour in a new image with the same dimensions
    _output = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)
    cv2.LUT(segmentation_ind_3ch, label_colours, _output)
    return _output


def softmax(x):
    '''Softmax function, transforming logits into probabilities'''
    out = np.exp(x)
    out = out / np.sum(out, axis=0)
    return out


def combineEnsemble(net_outputs, method, weighting):
    #This method will combine the output of models forming an Ensemble
    output = 0

    # If there is only one model, we skip this step
    if np.asarray(net_outputs).shape[0] == 1:
        return np.squeeze(net_outputs)

    if method == ensemble_pb2.VOTING:  # Majority voting
            # Calculates the label (by looking at the maximum class score)
            net_outputs = np.squeeze(np.asarray(net_outputs).argmax(axis=1))

            # Looks for the most common label for each pixel
            output = np.squeeze(stats.mode(net_outputs, axis=0)[0]).astype(int)

    if method == ensemble_pb2.LOGITARI:  # Logit arithmetic averaging
            output = np.zeros(net_outputs[0].shape)

            sum_weightings = 0
            for net_output, current_weighting in zip(net_outputs, weighting):
                output = output + net_output * current_weighting
                sum_weightings = sum_weightings + current_weighting

            # Make it a mean instead of a sum of logits
            output = output/sum_weightings

    if method == ensemble_pb2.LOGITGEO:  # Logit geometric averaging (NOT WORKING, BECAUSE LOGITS CAN BE NEGATIVE)
            output = np.ones(net_outputs[0].shape)

            for net_output in net_outputs:
                output = np.multiply(output, net_output)

            # Make it a mean
            output = np.power(output, float(1)/len(net_outputs))

    if method == ensemble_pb2.PROBAARI:  # Probability arithmetic averaging
        output = np.zeros(net_outputs[0].shape)

        sum_weightings = 0
        for net_output, current_weighting in zip(net_outputs, weighting):
            output = output + softmax(net_output) * current_weighting
            sum_weightings = sum_weightings + current_weighting

        # Make it a mean instead of a sum of probabilities
        output = output/sum_weightings

    if method == ensemble_pb2.PROBAGEO:  # Probability geometric averaging
        output = np.ones(net_outputs[0].shape)

        for net_output in net_outputs:
            output = np.multiply(output, softmax(net_output))

        # Make it a mean
        output = np.power(output, float(1)/len(net_outputs))
    return output


def getSections(array_length, num_sections):
    section_length = array_length / num_sections
    starts = [section_length * i for i in range(num_sections)]
    ends = starts[1:] + [array_length]
    return zip(starts, ends)


def cropAndSegment(input_image, num_sections, model, mean):
    '''
    This function segments the image patch by patch, then assembles the
    patches into a complete segmentation of the image.
    '''
    input_array = np.asarray(input_image)
    # Set output segmentation shape

    # model_logits = np.zeros((model.getNumClasses(), input_array.shape[0],
    #                          input_array.shape[1]))
    model_logits = np.zeros((model.getNumClasses(),) + input_array.shape[:2])
    vert_sections = getSections(input_array.shape[0], num_sections)
    horiz_sections = getSections(input_array.shape[1], num_sections)
    # print 'input array:', input_array.shape
    for h_start, h_end in vert_sections:
        for w_start, w_end in horiz_sections:
            # Crop image
            # print 'indices:', h_start, '-', h_end, 'and', w_start, '-', w_end
            cropped_array = input_array[h_start:h_end, w_start:w_end, :]
            # print 'cropped array:', cropped_array.shape
            cropped_image = Image.fromarray(cropped_array)
            # Why do we take [0] here???
            logits = model.segmentImage(cropped_image, mean)[0]
            # print 'logit shape:', logits.shape

            # Get segmentation with cropped image
            model_logits[:, h_start:h_end, w_start:w_end] = logits
    return model_logits


def computeEnsembleLogits(input_image, input_shape, models, logit_cols, crop):
    logits = []
    # new_shape is the shape of images that will be input to the network, after
    # cropping and resizing
    # TODO maybe don't need input_shape - do the commented instead
    # crop_shape = (np.asarray(np.asarray(input_image).shape) / crop)
    # new_shape = np.array([1, 3, crop_shape[0], crop_shape[1]])

    new_shape = None
    if config.input.resize:
        new_shape = input_shape
    else:
        crop_shape = (np.asarray(np.asarray(input_image).shape) / crop)
        new_shape = np.array([1, 3, crop_shape[0], crop_shape[1]])

    # compute the logits from each model
    model_logits = None
    start = time.time()
    for model in models:
        if crop > 1:
            model_logits = cropAndSegment(input_image, crop, model, mean)
        else:
            model_logits = model.segmentImage(input_image, mean, new_shape)
        logits.append(np.squeeze(model_logits))

    # Get the time after the network process
    runTime = time.time() - start

    # Import the logits saved in folders
    if len(logit_cols)>0:
        image_name = basename(os.path.splitext(image_path)[0])
    for logit_col in logit_cols:
        model_logits = logit_col.getImageLogits(image_name)
        logits.append(np.squeeze(model_logits))

    # Combine the logits from each source
    model_weights = [i.weight for i in models + logit_cols]
    logits = combineEnsemble(logits, config.ensemble_type, model_weights)
    return logits, runTime


def displayProbability(input_image, logits, prob_to_show):
    probs = softmax(logits)
    showProb = probs[prob_to_show]
    showProb = showProb / np.max(showProb)
    cv2.imshow("probability", showProb)


def displayOutput(label_colours, input_image, guessed_labels, true_labels,
                  resize, wait=False):
    # Transform the class labels into a segmented image
    guessed_image = colourSegment(guessed_labels, label_colours)
    newSize = (int(resize*input_image.shape[1]),
               int(resize*input_image.shape[0]))
    cv2.imshow("Input", cv2.resize(input_image, newSize))
    cv2.imshow("Output", cv2.resize(guessed_image, newSize))

    if true_labels is not None:
        # display the ground truth as well
        gt_image = np.array(true_labels)
        if len(gt_image.shape) == 2:
            # Convert the ground truth into a RGB array
            gt_image = colourSegment(gt_image, label_colours)
        elif len(gt_image.shape) != 3:
            print 'Unknown labels format'

        newSize = (int(resize*gt_image.shape[1]),
                   int(resize*gt_image.shape[0]))
        labeledImage = cv2.resize(gt_image, newSize)
        cv2.imshow("Ground truth", labeledImage)

    key = 0
    if wait:
        key = cv2.waitKey(0)
    else:
        key = cv2.waitKey(100)
    return key

def initialiseVideo(shape):
        fps = 12.0
        # Initialize video recording
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        print "Creating video with shape", shape
        images = cv2.VideoWriter(args.record + 'img.avi', fourcc, fps, shape)
        #labels = cv2.VideoWriter(args.record + 'labels.avi', fourcc, fps, shape)
        segmentation = cv2.VideoWriter(args.record + 'segmentation.avi',
                                       fourcc, fps, shape)
        return (images, segmentation)

def writeVideo(images, segmentation, input_image, guessed_labels, label_colours, resize):
        # Transform the class labels into a segmented image
        guessed_image = colourSegment(guessed_labels, label_colours)
        newSize = (int(resize*input_image.shape[1]),
               int(resize*input_image.shape[0]))
        images.write(cv2.resize(input_image, newSize))
        segmentation.write(cv2.resize(guessed_image, newSize))

        return (images, segmentation)

if __name__ == '__main__':
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

    # Load all networks and model output folders specified in .proto
    models = []
    for model_buf in config.model:
        models.append(Model(model_buf))
    logit_cols = []
    for logitFolderBuf in config.logitFolder:
        logit_cols.append(LogitCollection(logitFolderBuf))

    # get the various dimensions of the network
    input_shape = default_input_shape
    numb_cla = default_numb_cla
    if len(models) > 0:
        input_shape = models[0].getInputShape()
        numb_cla = models[0].getNumClasses()


    # Create the appropriate image iterator
    imageIterator = None
    if config.input.type == ensemble_pb2.Input.VIDEO:
        imageIterator = iterators.VideoIterator(config.input.file)
    elif config.input.type == ensemble_pb2.Input.IMAGES:
        imageIterator = iterators.FileListIterator(config.input.file)
    elif config.input.type == ensemble_pb2.Input.LABELS:
        # Histogram for evan's metrics
        totalHist = np.zeros((numb_cla, numb_cla))
        imageIterator = iterators.FileListIterator(config.input.file,
                                                   pairs=True)
    else:
        raise ValueError("No data provided in the prototxt!")

    # If an output folder is provided, create a list.txt file for summarising
    # which numpy matrices of which images have been created
    if config.outputFolder != "":
        summaryFile = open(config.outputFolder+"list.txt", 'w')

    mean = np.array([config.input.mean.r,
                     config.input.mean.g,
                     config.input.mean.b])

    # process each image, one-by-one
    n_im = 1  # Image counter
    times = []  # Variable for test times
    # Read the colours of the classes
    display = (args.record == '' and not args.hide)
    if display or args.record != '':
        label_colours = cv2.imread(config.input.colours).astype(np.uint8)

    # Initialization is finished, so begin iterating over the images.
    for input_image, real_label, image_path in imageIterator:

        #If we want to only keep the central part of a too big image
        if args.center>0.0 and args.center<1.0:
            w,h = input_image.size
            dw = int(args.center*w) #Cropped image size
            dh = int(args.center*h)
            a = int((w-dw)/2) #Crop coordinates
            b = int((h-dh)/2)
            input_image = input_image.crop((a, b, a+dw, b+dh))
            real_label = real_label.crop((a, b, a+dw, b+dh))
            print a,b,a+dw,b+dh

        #If we want to resize all inputs to the same size
        if config.input.resize:
            input_image = input_image.resize((input_shape[3], input_shape[2]),
                                             Image.ANTIALIAS)

        #Compute the logits (outputs) of the model(s)
        logits, runTime = computeEnsembleLogits(input_image, input_shape,
                                                models, logit_cols, args.crop)
        times.append(runTime)

        #Store eventually the path of the input image
        if image_path is not None:
                image_name = basename(os.path.splitext(image_path)[0])
        else:
                image_name = ""

        # TODO: Not sure of the first if, needed for making resnet output work
        if len(logits.shape) == 1 and len(logits[0].shape) == 4:
            logits = logits[0][0]
        elif len(logits.shape) == 1:
            logits = logits[0]
        # If we want to save network outputs to folder
        if config.outputFolder != "":
            np.save(config.outputFolder + image_name + ".npy", logits)
            #Save the stored outputs in a summary list file
            summaryFile.write(image_path + " " + config.outputFolder +
                              image_name + ".npy\n")



        # label each pixel as the class with the biggest logit
        if len(logits.shape) == 3:
            guessed_labels = logits.argmax(axis=0)
        elif len(logits.shape) != 2:
            print 'Unknown output shape:', logits.shape
            break

        #If ground-truth is provided, compute the metrics
        if real_label is not None:
            real_label = real_label.resize(input_image.size, Image.NEAREST)

            # If pascal VOC, reshape the label to HxWx1s
            np_real_label = np.array(real_label)
            if len(np_real_label.shape) == 3:
                real_label = Image.fromarray(np_real_label[:, :, 0])

            # Calculate the histogram for this image
            gtArray = np.array(real_label).flatten()
            guessArray = np.array(guessed_labels).flatten()
            hist = score.fast_hist(gtArray, guessArray, numb_cla)
            totalHist += hist

            if args.record != '' or args.hide:
                # Print image number and accuracy
                acc = score.computeSegmentationScores(hist).overallAcc
                print "Image", n_im, "(", image_name, ") accuracy:", acc


        # Switch to RGB (PIL Image read files as BGR)
        if len(np.array(input_image).shape) is 3:
            input_image = np.array(cv2.cvtColor(np.array(input_image),
                                                cv2.COLOR_BGR2RGB))

        #If we want to record a video
        if args.record != '':
                shape = (int(input_image.shape[1]*args.view_resize), int(input_image.shape[0]*args.view_resize))
                if n_im == 1: #If it's the first image, we initialise video
                        print "INITIALISE VIDEO TO INPUT IMAGE SHAPE: ", shape
                        images, segmentation = initialiseVideo(shape)
                print "Writing frame", n_im
                images, segmentation = writeVideo(images, segmentation, input_image, guessed_labels, label_colours, args.view_resize)


        # Display input and output
        if display:
            if args.show_prob is not None:
                displayProbability(input_image, logits, args.show_prob)
            key = displayOutput(label_colours, input_image, guessed_labels,
                                real_label, args.view_resize, args.key)
            if key % 256 == 27:  # exit on ESC - keycode is platform-dependent
                break

        n_im += 1

    # Exit properly
    if args.record != '':
        images.release()
        segmentation.release()
        #labels.release()
    elif not args.hide:
        cv2.destroyAllWindows()

    # If we opened an output file, close it
    if config.outputFolder != "":
        summaryFile.close()

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
