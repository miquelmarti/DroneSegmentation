# Convert each label files from HxWxC (C=3) to HxWxC (C=1)
# sudo python convert_labels.py ../shared/datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/train_tmp.txt pascal_voc_21_colors.png ../shared/datasets/VOCdevkit/VOC2012/SegmentationClass/

import lmdb
import argparse
import numpy as np
import png
import cv2
import sys
import os
from PIL import Image
import numpngw




parser = argparse.ArgumentParser()

# Mandatory options
parser.add_argument('text_file_with_paths', type=str, help='Path to the file that lists the absolute path to each image')
parser.add_argument('path_to_colors', type=str, help='Colors of the dataset, can be created with find_classes_colors.py')
parser.add_argument('path_output', type=str, help='Path where to create the new labels')

args = parser.parse_args()


colors = cv2.imread(args.path_to_colors)[0]



def getColorIndex(pixel):
    for i in range(0, len(colors)):
        if colors[i][0] == pixel[0] and colors[i][1] == pixel[1] and colors[i][2] == pixel[2]:
            return i
    
    print 'Found unknown color... Exit'
    print pixel
    print colors
    sys.exit()



for in_idx, in_ in enumerate(open(args.text_file_with_paths)):
    im = np.array(cv2.imread(in_.rstrip()))
    newImage = np.empty_like(im)
    newImage.resize((newImage.shape[0],newImage.shape[1],3))
    
    for i in range(0, im.shape[0]):
        for j in range(0, im.shape[1]):
            newImage[i][j] = [getColorIndex(im[i][j]), getColorIndex(im[i][j]), getColorIndex(im[i][j])]
    
    print 'img: ' + str(in_idx+1) + ' -> done'
    
    path, filename = os.path.split(in_.rstrip())
    newImage_ = Image.fromarray(newImage)
    newImage_.save(args.path_output + filename)


