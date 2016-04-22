# Convert each label files from colours to number of classes
# sudo python convert_labels.py /home/shared/datasets/VOCdevkit/VOC2012/SegmentationClass /home/shared/datasets/VOCdevkit/VOC2012/colours/pascal_voc_21_colours.png /home/shared/datasets/VOCdevkit/VOC2012/SegmentationClass_2/


import lmdb
import argparse
import numpy as np
import png
import cv2
import sys
import os
from PIL import Image
import numpngw



def get_arguments():
    # Import arguments
    parser = argparse.ArgumentParser()
    
    # Mandatory options
    parser.add_argument('path_input', type=str, \
                                    help='Path to the file that lists the absolute path to each image')
    parser.add_argument('path_to_transition', type=str, \
                                    help='File with the transition matrix (color.png or transition.png) ; created by create_transition.py')
    parser.add_argument('path_output', type=str, \
                                    help='Path where to create the new labels')
    
    return parser.parse_args()



def getColourIndex(pixel, transition):
    # Find the corresponding class
    for i in range(0, len(transition)):
        if transition[i][0] == pixel[0] and transition[i][1] == pixel[1] and transition[i][2] == pixel[2]:
            return i
    
    print 'Found unknown class... Exit'
    print pixel
    print transition
    sys.exit()



def main(args):
    
    # Get all options
    args = get_arguments()
    
    # Get the colours
    colours = cv2.imread(args.path_to_transition)[0]
    
    # For each label
    for in_idx, in_ in enumerate(os.listdir(args.path_input)):
        
        # Get the absolute path of the image
        full_path = os.path.join(args.path_input, in_.rstrip())
        
        # Get the image and create a copy
        im = np.array(cv2.imread(full_path))
        newImage = np.empty_like(im)
        newImage.resize((newImage.shape[0], newImage.shape[1], 3))
        
        # Replace each pixel of the copy by the number of the class
        for i in range(0, im.shape[0]):
            for j in range(0, im.shape[1]):
                col = getColourIndex(im[i][j], colours)
                newImage[i][j] = [col, col, col]
        
        print 'img: ' + str(in_idx+1) + ' -> done'
        
        # Save the new label
        newImage_ = Image.fromarray(newImage)
        newImage_.save(os.path.join(args.path_output, in_))
        



if __name__ == '__main__':
    
    main(None)
    
    pass


