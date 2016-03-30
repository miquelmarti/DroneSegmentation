# Return output.png, the file with all the colors in the database
# Need to be launch with a train.txt linking the labels images (with HxWxC C=3)
# EX : python create_drone_colors.py ../shared/datasets/Kai_drone_camVid/labels/lab.txt drone_colors.png

import caffe
import lmdb
import argparse
import numpy as np
import png
import cv2
import Image
import scipy.misc
    

parser = argparse.ArgumentParser()

# Mandatory options
parser.add_argument('text_file_with_paths', type=str, help='Path to the file that lists the absolute path to each image')
parser.add_argument('output', type=str, help='Name of the output.png')

arguments = parser.parse_args()


classes = [[[0,     0,      0],     # background
            [153,   153,    153],   # sky
            [255,   0,      0],     # house
            [229,   0,      0],     # building
            [204,   0,      0],     # airport
            [178,   0,      0],     # bridge
            [153,   0,      0],     # port
            [127,   0,      0],     # tent
            [229,   125,    0],     # dirt road
            [204,   111,    0],     # paved road
            [178,   97,     0],     # highway
            [153,   83,     0],     # railroad
            [285,   185,    102],   # parking area
            [255,   173,    76],    # mountain
            [229,   136,    25],    # square
            [0,     57,     255],   # water surface
            [102,   136,    255],   # beach
            [0,     40,     178],   # swimming pool
            [18,    255,    0],     # tree
            [13,    178,    0],     # grass surface
            [136,   255,    127],   # crops
            [255,   232,    0],     # car
            [178,   162,    0],     # bicycle
            [255,   241,    102],   # motor bike
            [255,   0,      213],   # passenger jetliner
            [255,   0,      213],   # private jet
            [255,   0,      213],   # propeller plane
            [255,   0,      213],   # fighter
            [255,   0,      213],   # millitary plane
            [255,   0,      213],   # airship
            [255,   0,      213],   # balloon
            [255,   0,      213],   # passenger balloon
            [255,   0,      213],   # helicopter
            [255,   0,      213],   # drone
            [255,   0,      213],   # passenger jetliner
            [255,   255,    255],   # human
            [255,   255,    255],   # dog
            [255,   255,    255],   # cat
            [255,   255,    255],   # monkey
            [255,   255,    255],   # elephant
            [255,   255,    255],   # cow
            [255,   255,    255],   # horse
            [255,   255,    255],   # rabbit
            [255,   255,    255],   # pig
            [255,   255,    255],   # lion
            [255,   255,    255],   # cheetah
            [255,   255,    255],   # panda
            [255,   255,    255],   # deer
            [0,     255,    255],   # eagle
            [0,     255,    255],   # falcon
            [0,     255,    255],   # dolphin
            [0,     255,    255],   # penguin
            [0,     255,    255],   # fishing boat
            [0,     255,    255],   # passenger ferry
            [0,     255,    255],   # cruise chip
            [0,     255,    255],   # millitary ship
            [0,     255,    255],   # boat
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [128,255,0]]]



def main(args):
    
    cpt = 1
    
    print classes
    img_new = Image.new('RGB', (len(classes[0]), len(classes)))
    img_new.putdata([tuple(p) for row in classes for p in row])
    img_new.save(arguments.output)




if __name__ == '__main__':
    
    main(None)
    
    pass
