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
    parser.add_argument('--output_lab', type=str, required=True, \
                                    help=   'Path to label output folder')
    parser.add_argument('--nb', type=int, required=True, \
                                    help=   'Number of images to create')
    
    return parser.parse_args()


def flip(image, label):
    if bool(np.random.choice(2)): # Horizontal flip
        image = image[:, ::-1, :]
        label = label[:, ::-1, :]
    if bool(np.random.choice(2)): # Vertical flip
        image = image[::-1, :, :]
        label = label[:, ::-1, :]
    return (image, label)

def random_crop(image, label, crop_size_0, crop_size_1):
    randX = np.random.choice(image.shape[0]-crop_size_0)
    randY = np.random.choice(image.shape[1]-crop_size_1)

    image = image[randX:randX+crop_size_0, randY:randY+crop_size_1]
    label = label[randX:randX+crop_size_0, randY:randY+crop_size_1]

    return (image, label)

def main(args):
     # Get all options
    args = get_arguments()

    crop_size_0 = 540
    crop_size_1 = 960
    
    listFile = open(args.input, 'r')


    output_img_directory = args.output_img
    if output_img_directory[len(output_img_directory)-1:len(output_img_directory)] is not '/':
        output_img_directory = output_img_directory + '/'

    output_lab_directory = args.output_lab
    if output_lab_directory[len(output_lab_directory)-1:len(output_lab_directory)] is not '/':
        output_lab_directory = output_lab_directory + '/'

    outputListFile = open(output_img_directory+"list.txt", 'w')

    number_dataset_iter = 0
    
    # For each image
    for nb_iter in range(0, args.nb):
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
        label_path = line.split()[1]

        
        # load image:
        img = cv2.imread(image_path)
        lab = cv2.imread(label_path)


        img_cropped, lab_cropped = random_crop(img, lab, crop_size_0, crop_size_1)
        image_out, label_out = flip(img_cropped, lab_cropped)
        

        # Write image

        image_name = os.path.basename(image_path)
        image_name = image_name[:len(image_name)-4]

        label_name = os.path.basename(label_path)
        label_name = label_name[:len(label_name)-4]

        image_extension = os.path.basename(image_path)[len(os.path.basename(image_path))-4:]
        label_extension = os.path.basename(label_path)[len(os.path.basename(label_path))-4:]

        output_filename_image = output_img_directory + image_name + '_' + str(number_dataset_iter) + image_extension
        output_filename_label = output_lab_directory + label_name + '_' + str(number_dataset_iter) + label_extension
        
        cv2.imwrite(output_filename_image,image_out)
        cv2.imwrite(output_filename_label,label_out)
        outputListFile.write(output_filename_image+'\t'+output_filename_label+'\n')
        
        #print output_filename_image
                        


if __name__ == '__main__':
    
    main(None)
    
    pass


