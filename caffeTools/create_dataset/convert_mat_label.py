# Resize images in the folder input_path
# python caffeTools/create_dataset/crop_resize_dataset.py --resize 1 --crop 8 --input_path /home/shared/data/datasets/droneye_dataset/JPEGImages --output_path /home/shared/data/datasets/droneye_dataset/JPEGImages_cut64


import numpy as np
import glob
from random import shuffle
import cv2
import os.path
import argparse
from PIL import Image

import scipy.io


def main(args):
     # Get all options
    
    listFile = open('/home/shared/data/datasets/SBD_dataset/dataset/train.txt', 'r')


    number_dataset_iter = 0
    
    # For each image
    while True:

        try:
            line = listFile.next()
            #break
        except StopIteration:
            break
        
        
        image_name = line.split()[0]

        
        # load image:
        mat = scipy.io.loadmat('/home/shared/data/datasets/SBD_dataset/dataset/cls/'+image_name+'.mat')
        label = mat['GTcls'][0]['Segmentation'][0].astype(np.uint8)
        label = label[np.newaxis, ...].squeeze()

        # Write image
        cv2.imwrite('/home/shared/data/datasets/SBD_dataset/dataset/SegmentationClass/'+image_name+'.png',label)
        
                        


if __name__ == '__main__':
    
    main(None)
    
    pass


