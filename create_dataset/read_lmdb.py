# This script reads images one by one from a LMDB database. It doesn't matter if the image is a RGB or label image, it will be read. Optionally, a colour code can be given to assign class labels into RGB values
# Example :
# python read_lmdb.py --file /home/shared/datasets/VOCdevkit/VOC2012/LMDB/train_lmdb/ --colours /home/shared/datasets/VOCdevkit/VOC2012/colors/pascal_voc_21_colors.png

#This code is partly taken from : http://stackoverflow.com/questions/33117607/caffe-reading-lmdb-from-python

import caffe
import lmdb
import numpy as np
import os
import caffe.proto.caffe_pb2
from caffe.io import datum_to_array
import cv2
import argparse

def get_arguments():
    # Import arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--file', type=str, required=True, \
                                    help=   'Path to the LMDB file folder (ex: /home/.../train_lmdb)')
    parser.add_argument('--colours', type=str, required=False, default='', \
                                    help=   'Colour code for class labels \
                                            (data/CamVid/colours/camvid12.png)')

    return parser.parse_args()

if __name__ == '__main__':

        # Get all options
        args = get_arguments()
        
        #Open LMDB
        lmdb_env = lmdb.open(args.file)
        lmdb_txn = lmdb_env.begin()
        lmdb_cursor = lmdb_txn.cursor()
        datum = caffe.proto.caffe_pb2.Datum()
        
        #If we specify colour code, open the .png file
        if args.colours != '':
                label_colours = cv2.imread(args.colours).astype(np.uint8)

        #Iterate through the images of the LMDB
        for key, value in lmdb_cursor:
            datum.ParseFromString(value)
            label = datum.label
            data = caffe.io.datum_to_array(datum) #The image will be stored in "data"
            
            data = np.transpose(data, (1,2,0)) #Transpose from C*H*W to H*W*C
            
            if(data.shape[2]==3): #If RGB image
                multiply=1
            else: #If label
                if args.colours != '': #If we want to use the colour code
                        #Reshape the data from H*W*1 to H*W*3 (needed for LUT function)
                        data = np.transpose(np.resize(np.transpose(data,(2,0,1)), (3, data.shape[0],data.shape[1])),(1,2,0))
                        output = np.zeros((data.shape[0], data.shape[1],3), dtype=np.uint8)
                        cv2.LUT(data, label_colours, output) #Colour code : replace class label by RGB code
                        data = output
                        multiply=1
                else:
                        multiply=12 #If no colour code, multiply label values by 12 so we still see something (with RGB values between 0 and 21 we only will see very dark colours, better have values between 0 and 256)
            
            cv2.imshow("Image",data*multiply) #Show the data
            key = cv2.waitKey(0)

            if key % 256 == 27: # exit on ESC - keycode is platform-dependent
                    break
