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
    
    parser.add_argument('--resize', type=int, required=True, \
                                    help=   'Divide image size by this factor')
    parser.add_argument('--crop', type=int, required=True, \
                                    help=   'Crop image into crop*crop small images')
    parser.add_argument('--input_path', type=str, required=True, \
                                    help=   'Path to the folder with all images to resize')
    parser.add_argument('--output_path', type=str, required=True, \
                                    help=   'Path to output folder')
    parser.add_argument('--labels', type=bool, default=False, \
                                    help=   'Specify if dealing with labels (different interpolation for labels and RGB)')
    return parser.parse_args()


def main(args):
     # Get all options
    args = get_arguments()
    
    # For each image
    for in_idx, in_ in enumerate(os.listdir(args.input_path)):
        
        # load image:
        full_path = os.path.join(args.input_path, in_.rstrip())
        img = cv2.imread(full_path)
        
        crop_number = 0
        for id_h in range(0,args.crop):
                crop_h = int(img.shape[0]/float(args.crop))
                h1 = id_h*crop_h
                
                
                for id_w in range(0,args.crop+0):
                        crop_w = int(img.shape[1]/float(args.crop))
                        w1 = id_w*crop_w
                        
                        cropped = img[h1:(h1+crop_h), w1:(w1+crop_w)]
        
                        if args.resize > 1:
                                cropped = Image.fromarray(cropped)
                                # Resize mode should be different for labels (solid values)
                                if args.labels:
                                    cropped = cropped.resize([int(crop_w+1/float(args.resize)), int(crop_h+1/float(args.resize))],Image.NEAREST)
                                else:
                                    cropped = cropped.resize([int(crop_w+1/float(args.resize)), int(crop_h+1/float(args.resize))], Image.ANTIALIAS)
                                
                                cropped = np.asarray(cropped)
                                

        
                        # Write image
                        output_filename = os.path.join(args.output_path, in_.rstrip())
                        output_filename = output_filename[:len(output_filename)-4] + "_" + str(crop_number) + output_filename[len(output_filename)-4:]
                        
                        cv2.imwrite(output_filename,cropped)
                        
                        print output_filename
                        crop_number += 1
                        
        #print 'img: ' + str(in_idx+1) + ' -> done'


if __name__ == '__main__':
    
    main(None)
    
    pass


