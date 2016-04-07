# Resize images in the file_with_paths
# python resize_dataset.py --W 480 --H 360 --output_path /home/johannes/test/ --input_images /home/shared/datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/train_img.txt --labels False


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
    parser.add_argument('--output_path', type=str, required=True, \
                                    help=   'Path to output LMDB folder')
    parser.add_argument('--input_images', type=str, required=True, \
                                    help=   'Path to .txt file with all images to store in the LMDB database')
    parser.add_argument('--labels', type=bool, default=False, \
                                    help=   'Precise if dealing with labels (different interpolation for labels and RGB)')
    return parser.parse_args()


def main(args):
 	# Get all options
    	args = get_arguments()

	for in_idx, in_ in enumerate(open(args.input_images)):
		print 'Loading image ', in_.rstrip()

		# load image:
		img = cv2.imread(in_.rstrip())
		img = Image.fromarray(img)

		if args.labels:
		    img = img.resize([args.W, args.H],Image.NEAREST)

		else:
		    img = img.resize([args.W, args.H], Image.ANTIALIAS)

		output_filename = args.output_path + os.path.basename(in_.rstrip())
		print 'WRITE IN ', output_filename
		cv2.imwrite(output_filename,np.asarray(img))


if __name__ == '__main__':
    
    main(None)
    
    pass


