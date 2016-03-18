import numpy as np
import matplotlib.pyplot as plt
import os.path
import scipy
import argparse
import math
import cv2
import sys
import time


#sys.path.append('/usr/local/lib/python2.7/site-packages')
# Make sure that caffe is on the python path:
#caffe_root = '/home/shared/caffeFcn/'
#sys.path.insert(0, caffe_root + 'python')
import caffe


def segmentImage(image):
    start = time.time()

    #Resize input image
    image = cv2.resize(image, (input_shape[3],input_shape[2]))
    input_image = image.transpose((2,0,1))
    input_image = input_image[(2,1,0),:,:]
    input_image = np.asarray([input_image])
    end = time.time()
    print '%30s' % 'Resized image in ', str((end - start)*1000), 'ms'

    #Feed forward and calculate the network
    start = time.time()
    out = net.forward_all(data=input_image)
    end = time.time()
    print '%30s' % 'Executed FCN-8 in ', str((end - start)*1000), 'ms'

    #Get the last datablob and process it
    start = time.time()
    segmentation_ind = np.squeeze(net.blobs['upscore'].data) #Size : 21x500x500 with class scores

    #Argmax : choose the best index
    segmentation_ind_3ch = np.argmax(segmentation_ind, axis=0)

    #Change the shape of segmentated so we can use it for look-up table
    segmentation_ind_3ch = np.resize(segmentation_ind_3ch,(3,input_shape[2],input_shape[3]))
    segmentation_ind_3ch = segmentation_ind_3ch.transpose(1,2,0).astype(np.uint8)
    segmentation_rgb = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)


    #Look up table to assign colours to each output
    cv2.LUT(segmentation_ind_3ch,label_colours,segmentation_rgb)
    segmentation_rgb = segmentation_rgb.astype(float)/255

    end = time.time()
    print '%30s' % 'Processed results in ', str((end - start)*1000), 'ms\n'



    #Plotting
    cv2.imshow("Input", image)
    cv2.imshow("FCN-8", segmentation_rgb)

    return segmentation_rgb


if __name__ == "__main__":
    # Import arguments
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--video', type=str) #If input is video (ex .mp4)
    group.add_argument('--image', type=str) #If input is single image
    parser.add_argument('--model', type=str, required=True) #.prototxt file for architecture
    parser.add_argument('--weights', type=str, required=True) #.caffemodel file for the weights
    parser.add_argument('--colours', type=str, required=True) #Colour lookup table
    args = parser.parse_args()

    #Creates the network
    net = caffe.Net(args.model,
                    args.weights,
                    caffe.TEST)

    #Set GPU mode (possibility to run on CPU with .set_mode_cpu() )
    caffe.set_mode_gpu()

    #Shapes of input/output
    input_shape = net.blobs['data'].data.shape
    output_shape = net.blobs['upscore'].data.shape

    #Read the look-up table image to colour segmented parts
    label_colours = cv2.imread(args.colours).astype(np.uint8)


    cv2.namedWindow("Input")
    cv2.namedWindow("FCN-8")

    if args.image is not None:
        image = cv2.imread(args.image)
        segmentation_rgb = segmentImage(image)

        cv2.waitKey(0)

#If no input arguments, we use the default image sequence
    else: 
        if args.video is not None:
            cap = cv2.VideoCapture(args.video)
        else:
            cap = cv2.VideoCapture("/home/shared/datasets/CamVid/test2/%03d.png") # Change this to your webcam ID, or file name for your video file


        if cap.isOpened(): # try to get the first frame
            rval, frame = cap.read()
        else:
            rval = False

        while rval:
            start = time.time()
            rval, frame = cap.read()
            end = time.time()
            print '%30s' % 'Grabbed camera frame in ', str((end - start)*1000), 'ms'

            segmentation_rgb = segmentImage(frame)

            key = cv2.waitKey(1)
            if key == 27: # exit on ESC
                break

    cap.release()
    cv2.destroyAllWindows()

