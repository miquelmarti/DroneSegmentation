'''
RUN COMMAND EXAMPLE :
python view_weights.py --model /home/johannes/project/models/segnet/deploy-pascal.prototxt --weights /home/johannes/project/models/segnet/train_pascal/ --layer "conv1_1" --scale_divide 0.5 --scale_shift 0.0

Press ESC to exit
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image
import argparse
import time
import cv2
import os

os.environ['GLOG_minloglevel'] = '2' #This disables the text written automatically by the Caffe C++ module
import caffe

def get_arguments():
    # Import arguments
    parser = argparse.ArgumentParser()
    
    # Mandatory options
    parser.add_argument('--model', type=str, required=True, help='Path to the .prototxt model')
    parser.add_argument('--weights', type=str, required=True, help='Path to the .caffemodel weights, or a path to a folder containing multiple caffemodels')
    parser.add_argument('--layer', type=str, required=True, help='Name of the layer to visualise (the name as seen in the prototxt)')
    
    # Optional options
    parser.add_argument('--cpu', type=bool, required=False, default=False, help='Default False, set True for CPU mode')
    parser.add_argument('--terminal', type=bool, required=False, default=False, help='Set true for printing weights in terminal')
    parser.add_argument('--scale_shift', type=float, required=False, default=0.0, help='Substracts the data value by this before plot')
    parser.add_argument('--scale_divide', type=float, required=False, default=1.0, help='Divides the data value by this before plot')
    
    return parser.parse_args()


def build_network(args):
    # GPU / CPU mode
    if args.cpu:
        print 'Set CPU mode'
        #caffe.set_mode_cpu()
    else:
        print 'Set GPU mode'
        caffe.set_mode_gpu()
    
    # Creation of the network
    net = caffe.Net(args.model,      # defines the structure of the model
                        args.weights,    # contains the trained weights
                        caffe.TEST)      # use test mode (e.g., don't perform dropout)
    return net

def vis_square(data, scale_shift, scale_divide):
    """Code by Caffe. 
    Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    if data.max()-data.min()>0:
            
            #data = (data - data.min()) / (data.max() - data.min())
            data = (data-scale_shift)/scale_divide
            
            # force the number of filters to be square
            n = int(np.ceil(np.sqrt(data.shape[0])))
            padding = (((0, n ** 2 - data.shape[0]),
                       (0, 1), (0, 1))                 # add some space between filters
                       + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
            data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
            
            # tile the filters into an image
            data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
            data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
            
            #Plot a 600x600 window of the weights
            cv2.imshow("Data", cv2.resize(data,(600,600),0,0,0,cv2.INTER_NEAREST))
            
    else: 
        print "Filter weights are all equal to ", data.max(), " !"
    
    
def main(args, caffemodel):
    print "Opening ", caffemodel
    # Set the network according to the arguments
    net = build_network(args)

    # Which layer to see ?
    layer_to_see = args.layer

    # Extract weights
    weights = net.params[layer_to_see][0].data
    if args.terminal:
    	print "WEIGHTS of layer", layer_to_see, " (of shape ", weights.shape ,") : ", weights
    #Weight format :  weights[filter_number_to_show,channel_to_show,:,:]

    # Extract bias
    bias = net.params[layer_to_see][1].data
    if args.terminal:
    	print "BIAS (of shape ", bias.shape ,") : ", bias
    	
    #Plot all weights in a window
    vis_square(weights.transpose(0, 2, 3, 1), args.scale_shift, args.scale_divide)

        
def getfiles(dirpath):
    #If we precised the path without slash at the end of directory name, add it.
    if dirpath.endswith("/"):
        extra=""
    else:
        extra="/"
   
    #Extracts files of folderin creation order. Source : http://stackoverflow.com/questions/168409/how-do-you-get-a-directory-listing-sorted-by-creation-date-in-python/168435#168435
    list_files = [dirpath+extra+file_name for file_name in os.listdir(dirpath)
         if os.path.isfile(os.path.join(dirpath, file_name))]
    list_files.sort(key=lambda file_name: os.path.getmtime(os.path.join(dirpath, file_name)))
    return list_files
    
if __name__ == '__main__':

    # Get all options
    args = get_arguments()
    
    if os.path.isdir(args.weights): #If we give a weight directory, iterate through files inside it
        list_files = getfiles(args.weights)
        for i in list_files:
            if i.endswith(".caffemodel"): #Only select caffemodel files in the folder
                main(args, i) #Run script
                
                key = cv2.waitKey(100)
                if key % 256 == 27: # exit on ESC
                        break
                continue
            else:
                continue


    else:
        #just take single weights file
        caffemodel = args.weights
        #Run the script
        main(args, caffemodel)
