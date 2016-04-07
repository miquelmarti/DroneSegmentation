# Resize images in the file_with_paths
# python resize_dataset.py

from PIL import Image
import numpy as np
import glob
from random import shuffle
import cv2
import os.path

W = 500 # Required Height
H = 500 # Required Width
file_with_paths = '/home/shared/datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/train_lab_classes.txt' # Text file with all the paths
output_folder = '/home/shared/datasets/VOCdevkit/VOC2012/SegmentationClass_500x500_classes/'
labels = True # Set to true if ground truth


for in_idx, in_ in enumerate(open(file_with_paths)):
	print 'Loading image ', in_.rstrip()

	# load image:
	img = cv2.imread(in_.rstrip())
	img = Image.fromarray(img)

	if labels:
	    img = img.resize([W, H],Image.NEAREST)

	else:
	    img = img.resize([W, H], Image.ANTIALIAS)

	output_filename = output_folder + os.path.basename(in_.rstrip())
	print 'WRITE IN ', output_filename
	cv2.imwrite(output_filename,np.asarray(img))
