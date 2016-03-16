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
    
    # Optional options
    parser.add_argument('--colours', type=str, required=False, default='', help='If the colours of the classes are provided (data/CamVid/colours/camvid12.png)')
    parser.add_argument('--gpu', type=bool, required=False, default=True, help='Default true, set False for CPU mode')
    parser.add_argument('--test', type=bool, required=False, default=True, help='Default true, set False for training mode')
    parser.add_argument('--input', type=str, required=False, default='data', help='Default is data, change it for the name of the input in your network (mostly foundable as the bottom of the first layer in the model (prototxt))')
    parser.add_argument('--output', type=str, required=False, default='output', help='Default is output, change it for the name of the output in your network (mostly foundable as the top of the last layer in the model (prototxt))')

    parser.add_argument('--labelled', type=str, required=False, default='', help='Path to the folder with labelled data ([...]/)')
    
    # Mandatory arguments
    parser.add_argument('data_to_segment', type=str, help='Path to the folder ([...]/)')
    
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


def mean_IU(guess,real):
	class_ranges = range(min(guess.min(),real.min()),max(guess.max(),real.max())+1)
	number_classes = 0
	IU_value = 0.0
	
	for ccc in class_ranges:
		true_positive  = sum(sum((real == ccc) & (guess == ccc)))
		false_positive = sum(sum((real != ccc) & (guess == ccc)))
		false_negative = sum(sum((real == ccc) & (guess != ccc)))
		
		if true_positive+false_positive+false_negative>0:
			IU_value = IU_value + true_positive/float(true_positive+false_positive+false_negative)
			#print 'TP: ', true_positive, ' FP: ', false_positive, ' FN: ', false_negative
			#print(true_positive/(true_positive+false_positive+false_negative))
			number_classes = number_classes+1
	
	result = 1/number_classes
	result = result*IU_value


	return IU_value/number_classes

	


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
    """
    # Given an Image, convert to ndarray and preprocess for VGG
    _input = Image.open(args.data_to_segment)
    frame = pre_processing(_input, input_blob.data.shape)
    
    # Shape for input (data blob is N x C x H x W), set data
    input_blob.reshape(1, *frame.shape)
    input_blob.data[...] = frame
    
    # Run the network and take argmax for prediction
    net.forward()
    
    # Get the final output
    # Remove single dimensional entries (was (1, 1, 360, 480), become (360, 480))
    #segmentation_ind = np.squeeze(output_blob.data)
    # Resize it for 3 channels, now (3, 360, 480)
    #segmentation_ind_3ch = np.resize(segmentation_ind,(3,input_shape[2],input_shape[3]))
    # Converts it to format H x W x C (was C x H x W)
    #segmentation_ind = output_blob.data.transpose(0,2,3,1).astype(np.uint8)
    #print segmentation_ind.shape
    # Create a new array (all zeros) with the shape of segmentation_ind_3ch
    #segmentation_rgb = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)
    _output = output_blob.data[0].argmax(axis=0)
    _output = _output.astype(float)/255
    
    
    # Display input and output
    print type(_output)
    plt.imshow(np.array(_input))
    plt.imshow(_output)
    plt.show()
    key = cv2.waitKey(1)
    time.sleep(100)
    """
    
    for filename in os.listdir(args.data_to_segment):
        # Given an Image, convert to ndarray and preprocess for VGG
        _input = Image.open(os.path.join(args.data_to_segment, filename))
        frame = pre_processing(_input, input_blob.data.shape)
        
        # Shape for input (data blob is N x C x H x W), set data
        input_blob.reshape(1, *frame.shape)
        input_blob.data[...] = frame
        
        # Run the network and take argmax for prediction
        net.forward()
        
        # Get the final output
        # Remove single dimensional entries (was (1, 1, 360, 480), become (360, 480))
        #segmentation_ind = np.squeeze(output_blob.data)
        # Resize it for 3 channels, now (3, 360, 480)
        #segmentation_ind_3ch = np.resize(segmentation_ind,(3,input_shape[2],input_shape[3]))
        # Converts it to format H x W x C (was C x H x W)
        #segmentation_ind = output_blob.data.transpose(0,2,3,1).astype(np.uint8)
        #print segmentation_ind.shape
        # Create a new array (all zeros) with the shape of segmentation_ind_3ch
        #segmentation_rgb = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)
        #_output = output_blob.data[0].argmax(axis=0)
        #_output = _output.astype(float)/255
        
        
        _output = 0

	guessed_labels = 0
        
        # If SegNet
        if args.colours != '':
            # Read the colours of the classes
            label_colours = cv2.imread(args.colours).astype(np.uint8)
            # Remove single dimensional entries (was (1, 1, 360, 480), become (360, 480))
            segmentation_ind = np.squeeze(output_blob.data)
            # Resize it for 3 channels, now (3, 360, 480)
            segmentation_ind_3ch = np.resize(segmentation_ind,(3,input_blob.data.shape[2],input_blob.data.shape[3]))
            # Converts it to format H x W x C (was C x H x W)
            segmentation_ind_3ch = segmentation_ind_3ch.transpose(1,2,0).astype(np.uint8)
            # Create a new array (all zeros) with the shape of segmentation_ind_3ch
            _output = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)

	    guessed_labels = segmentation_ind

            # Fill it with colours of classes
            cv2.LUT(segmentation_ind_3ch,label_colours,_output)
            _output = _output.astype(float)/255 # Optional, for convention
        
        # If FCN-8
        else:
            _output = output_blob.data[0].argmax(axis=0)
	    guessed_labels = _output
            palette_base = [i for i in xrange(0, 256, 255 / 3)]
            palette = [(palette_base[i], palette_base[j] , palette_base[k]) for i in xrange(4) for j in xrange(4) for k in xrange(4)]
            _output = np.array(palette, dtype=np.uint8)[_output]
        
	#If specify the ground-truth, calculate mean IU
	if args.labelled != '':
		real_label = np.array(Image.open(os.path.join(args.labelled, filename))) #Load labelled image
		#Calculate and print the mean IU
		print 'Mean IU : ', mean_IU(np.array(guessed_labels, dtype=np.uint8),np.array(real_label, dtype=np.uint8))

		#Tweeking so we can show the labelled image
		segmentation_ind_3ch = np.resize(real_label,(3,input_blob.data.shape[2],input_blob.data.shape[3]))
		segmentation_ind_3ch = segmentation_ind_3ch.transpose(1,2,0).astype(np.uint8)
		show_label = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)
		label_colours = cv2.imread(args.colours).astype(np.uint8)
		cv2.LUT(segmentation_ind_3ch,label_colours,show_label)
		show_label = show_label.astype(float)/255
		cv2.imshow("Labelled", show_label)
        
        # Display input and output
        cv2.imshow("Input", np.array(_input))
        cv2.imshow("Output", _output)
        
        key = cv2.waitKey(1)
        if key == 27: # exit on ESC
            break

    cv2.destroyAllWindows()
    


