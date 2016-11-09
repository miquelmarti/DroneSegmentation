# Return output.png, the file with all the colors / transitions of the database.
# python create_transition.py /home/shared/datasets/VOCdevkit/VOC2012/colours/colours.txt /home/shared/datasets/VOCdevkit/VOC2012/colours/pascal_voc_21_colours.png

import caffe
import lmdb
import argparse
import numpy as np
import png
import cv2
import Image
import scipy.misc



def get_arguments():
    # Import arguments
    parser = argparse.ArgumentParser()
    
    # Mandatory options
    parser.add_argument('input_transition', type=str, \
                                    help='Path to the file that lists the transitions')
    parser.add_argument('output', type=str, \
                                    help='Name of the output.png')
    
    return parser.parse_args()



def main(args):
    
    # Get all options
    args = get_arguments()
    
    # Get the transition file
    with open(args.input_transition) as f:
        classes = [[]]
        for line in f:
            line = line.split()[:3]
            if line:
                line = [int(i) for i in line]
                classes[0].append(line)
    
    # Complete the array to fit the PNG format
    while len(classes[0]) < 255:
        classes[0].append([0,0,0])
    
    # Save the png
    img_new = Image.new('RGB', (len(classes[0]), len(classes)))
    img_new.putdata([tuple(p) for row in classes for p in row])
    img_new.save(args.output)



if __name__ == '__main__':
    
    main(None)
    
    pass


