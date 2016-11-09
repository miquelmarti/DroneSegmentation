# Resize images in the folder input_path
# python caffeTools/create_dataset/crop_resize_dataset.py --resize 1 --crop 8 --input_path /home/shared/data/datasets/droneye_dataset/JPEGImages --output_path /home/shared/data/datasets/droneye_dataset/JPEGImages_cut64


import numpy as np
import glob
from random import shuffle
import cv2
import os.path
import argparse
from PIL import Image


def get_arguments():
    # Import arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input', type=str, required=True, \
                                    help=   'Path to input files')
    parser.add_argument('--output_img', type=str, required=True, \
                                    help=   'Path to image output folder')
    
    return parser.parse_args()



def main(args):
     # Get all options
    args = get_arguments()
    
    listFile = open(args.input, 'r')


    output_img_directory = args.output_img
    if output_img_directory[len(output_img_directory)-1:len(output_img_directory)] is not '/':
        output_img_directory = output_img_directory + '/'


    outputListFile = open(output_img_directory+"list.txt", 'w')

    number_dataset_iter = 0
    
    # For each image
    for nb_iter in range(0, 4*416):
        print nb_iter

        try:
            line = listFile.next()
            #break
        except StopIteration:
            listFile = open(args.input, 'r')   
            line = listFile.next()
            number_dataset_iter += 1
            print "Back to beginning of dataset"
        
        
        image_path = line.split()[0]

        
        # load image:
        image = cv2.imread(image_path)

        if number_dataset_iter==1 or number_dataset_iter==3: # Horizontal flip
            image = image[:, ::-1, :]
        if number_dataset_iter==2 or number_dataset_iter==3: # Vertical flip
            image = image[::-1, :, :]
        

        # Write image

        image_name = os.path.basename(image_path)
        image_name = image_name[:len(image_name)-4]

        image_extension = os.path.basename(image_path)[len(os.path.basename(image_path))-4:]

        output_filename_image = output_img_directory + image_name + '_' + str(number_dataset_iter) + image_extension
        
        cv2.imwrite(output_filename_image,image)
        outputListFile.write(output_filename_image+'\n')
        
                        


if __name__ == '__main__':
    
    main(None)
    
    pass


