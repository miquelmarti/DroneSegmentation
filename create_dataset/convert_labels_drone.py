# Convert each label files from HxWxC (C=3) to HxWxC (C=1)
# sudo python convert_labels_drone.py /home/shared/datasets/Kai_drone/labels/lab.txt drone_colors.png /home/shared/datasets/Kai_drone/labels_classes/


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

arguments = parser.parse_args()


colors = cv2.imread(arguments.path_to_colors)[0]
conversion = [  [0,0,0],
                [0,0,1],
                [0,1,0],
                [0,1,2],
                [0,1,3],
                [0,1,4],
                [0,1,5],
                [0,1,6],
                [0,2,0],
                [0,2,2],
                [0,2,3],
                [0,2,4],
                [0,2,5],
                [0,2,6],
                [0,2,7],
                [0,3,0],
                [0,3,1],
                [0,3,2],
                [0,4,0],
                [0,4,1],
                [0,4,2],
                [0,5,0],
                [0,5,1],
                [0,5,2],
                [0,6,0],
                [0,6,1],
                [0,6,2],
                [0,6,3],
                [0,6,4],
                [0,6,5],
                [0,6,6],
                [0,6,7],
                [0,6,8],
                [0,6,9],
                [0,7,0],
                [0,7,1],
                [0,7,2],
                [0,7,3],
                [0,7,4],
                [0,7,5],
                [0,7,6],
                [0,7,7],
                [0,7,8],
                [0,7,9],
                [0,7,10],
                [0,7,11],
                [0,7,12],
                [0,8,0],
                [0,8,1],
                [0,9,0],
                [0,9,1],
                [0,10,0],
                [0,10,1],
                [0,10,2],
                [0,10,3],
                [0,10,4]]






def getColorIndex(pixel):
    for i in range(0, len(conversion)):
        if conversion[i][0] == pixel[0] and conversion[i][1] == pixel[1] and conversion[i][2] == pixel[2]:
            return i
    
    return 255
    print 'Found unknown color... Exit'
    print pixel
    print conversion
    sys.exit()



def main(args):
    
    for in_idx, in_ in enumerate(open(arguments.text_file_with_paths)):
        im = np.array(cv2.imread(in_.rstrip()))
        IM = Image.open(in_.rstrip())
        im2 = np.array(IM)
        newImage = np.empty_like(im2)
        newImage.resize((newImage.shape[0], newImage.shape[1],3))
        
        for i in range(0, im2.shape[0]):
            for j in range(0, im2.shape[1]):
                col = getColorIndex(im2[i][j])
                newImage[i][j] = [col, col, col]
        
        print 'img: ' + str(in_idx+1) + ' -> done'
        
        path, filename = os.path.split(in_.rstrip())
        newImage_ = Image.fromarray(newImage)
        newImage_.save(arguments.path_output + filename)




if __name__ == '__main__':
    
    main(None)
    
    pass


