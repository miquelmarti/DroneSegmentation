# Return output.png, the file with all the colors in the database
# Need to be launch with a train.txt linking the labels images (with HxWxC C=3)
# EX : python find_classes_colors.py ../shared/datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/train_tmp.txt 21 pascal_voc_21_colors.png

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
parser.add_argument('number_of_classes', type=int, help='Number of classes of the datasets')
parser.add_argument('output', type=str, help='Name of the output.png')

args = parser.parse_args()


classes = [[[0,0,0]]]


def new_class(pixel):
    for i in range(0, len(classes[0])):
        if classes[0][i][0] == pixel[0] and classes[0][i][1] == pixel[1] and classes[0][i][2] == pixel[2]:
            return False
    
    return True

def toBGR(pixel):
    tmp = pixel[0]
    pixel[0] = pixel[2]
    pixel[2] = tmp
    return pixel



def main(args):
    
    for in_idx, in_ in enumerate(open(args.text_file_with_paths)):
        im = np.array(cv2.imread(in_.rstrip()))
        print 'img: ' + str(in_idx)
        
        for i in range(0, im.shape[0]):
            for j in range(0, im.shape[1]):
                if new_class(im[i][j]):
                    classes[0].append(im[i][j].tolist())
                    print str(len(classes[0])-1) + ' classes found'
        
        if len(classes[0]) > args.number_of_classes:
            break

    print classes
    for i in range(0, len(classes[0])):
        classes[0][i] = toBGR(classes[0][i])
    for i in range(len(classes[0]), 256):
        classes[0].append([0,0,0])


    img_new = Image.new('RGB', (len(classes[0]), len(classes)))
    img_new.putdata([tuple(p) for row in classes for p in row])
    img_new.save(args.output)




if __name__ == '__main__':
    
    main(None)
    
    pass
