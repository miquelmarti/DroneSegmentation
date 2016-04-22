# Return the colours of a dataset with specifics labels (colours with cv2, classes with PIL.Image). Works well for pascal VOC.
# python find_classes_colours.py /home/shared/datasets/VOCdevkit/VOC2012/unused/original_labels 21 /home/shared/datasets/VOCdevkit/VOC2012/colours/colours.txt

import caffe
import lmdb
import argparse
import numpy as np
import png
import cv2
import os
import Image
import scipy.misc


def get_arguments():
    # Import arguments
    parser = argparse.ArgumentParser()
    
    # Mandatory options
    parser.add_argument('input', type=str, \
                                    help='Path to the folder with all the labels')
    parser.add_argument('nb_of_classes', type=int, \
                                    help='Number of classes of the dataset')
    parser.add_argument('output', type=str, \
                                    help='Name of the output.txt')
    
    return parser.parse_args()



# Check if pixel is already present in classes
def new_class(pixel, classes):
    for i in range(0, len(classes)):
        if classes[i][0] == pixel[0] and classes[i][1] == pixel[1] and classes[i][2] == pixel[2]:
            return False
    
    return True


def main(args):
    
    # Get all options
    args = get_arguments()
    
    # List of the classes
    classes = [[-1]*3]*args.nb_of_classes
    
    # Counter of the founded classes
    cpt = 0
    
    # For each label
    for in_idx, in_ in enumerate(os.listdir(args.input)):
        
        # Get the absolute path of the image
        full_path = os.path.join(args.input, in_.rstrip())
        
        # Get the image
        im = np.array(cv2.imread(full_path))
        IM = np.array(Image.open(full_path), dtype=np.int)
        print 'img: ' + str(in_idx)
        
        # Get the new classes if there are some in this image
        for i in range(0, im.shape[0]):
            for j in range(0, im.shape[1]):
                if new_class(im[i][j], classes) and IM[i][j] < len(classes):
                    classes[IM[i][j]] = im[i][j].tolist()
                    cpt += 1
                    print str(cpt) + ' classes found'
        
        # If we found all the classes
        if cpt == args.nb_of_classes:
            break
        
    
    # Create the output
    outp_txt = ''
    for i in range(0, len(classes)):
        outp_txt += str(classes[i][0]) + '\t' + str(classes[i][1]) + '\t' + str(classes[i][2]) + '\t\t\t# \n'
    f = open(args.output, 'w')
    f.write(outp_txt.expandtabs(4))
    f.close()



if __name__ == '__main__':
    
    main(None)
    
    pass


