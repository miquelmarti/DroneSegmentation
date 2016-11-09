# Resize images in the folder input_path
# python resize_dataset.py --W 480 --H 360 --input_path /home/shared/datasets/VOCdevkit/VOC2012/JPEGImages --output_path /home/johannes/test_480x360/ --labels False


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
    
    parser.add_argument('--W', type=int, required=True, \
                                    help=   'Resize image to this width')
    parser.add_argument('--H', type=int, required=True, \
                                    help=   'Resize image to this height')
    parser.add_argument('--input_path', type=str, required=True, \
                                    help=   'Path to the folder with all images to resize')
    parser.add_argument('--output_path', type=str, required=True, \
                                    help=   'Path to output folder')
    parser.add_argument('--labels', type=bool, default=False, \
                                    help=   'Precise if dealing with labels (different interpolation for labels and RGB)')
    return parser.parse_args()


def main(args):
     # Get all options
    args = get_arguments()
    
    # For each image
    for in_idx, in_ in enumerate(os.listdir(args.input_path)):
        
        # load image:
        full_path = os.path.join(args.input_path, in_.rstrip())
        img = cv2.imread(full_path)
        img = Image.fromarray(img)
        
        # Resize mode should be different for labels (solid values)
        if args.labels:
            img = img.resize([args.W, args.H],Image.NEAREST)
        else:
            img = img.resize([args.W, args.H], Image.ANTIALIAS)
        
        # Write image
        output_filename = os.path.join(args.output_path, in_.rstrip())
        cv2.imwrite(output_filename,np.asarray(img))
        print 'img: ' + str(in_idx+1) + ' -> done'


if __name__ == '__main__':
    
    main(None)
    
    pass


