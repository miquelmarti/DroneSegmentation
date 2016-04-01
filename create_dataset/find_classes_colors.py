# Return output.png, the file with all the colors in the database. Need to be launch with a train.txt linking the labels images (with HxWxC C=3)
# python find_classes_colors.py /home/shared/datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/train_img.txt 21 pascal_voc_21_colors.png

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

arguments = parser.parse_args()


classes = [[]]
for i in range(0, 256):
    classes[0].append([-1,-1,-1])


def new_class(pixel):
    for i in range(0, len(classes[0])):
        if classes[0][i][0] == pixel[0] and classes[0][i][1] == pixel[1] and classes[0][i][2] == pixel[2]:
            return False
    
    return True



def main(args):
    
    cpt = 1
    
    for in_idx, in_ in enumerate(open(arguments.text_file_with_paths)):
        im = np.array(cv2.imread(in_.rstrip()))
        IM = np.array(Image.open(in_.rstrip()), dtype=np.int)
        print 'img: ' + str(in_idx)
        
        for i in range(0, im.shape[0]):
            for j in range(0, im.shape[1]):
                if new_class(im[i][j]):
                    classes[0][IM[i][j]][0] = im[i][j][0]
                    classes[0][IM[i][j]][1] = im[i][j][1]
                    classes[0][IM[i][j]][2] = im[i][j][2]
                    cpt += 1
                    print str(cpt-1) + ' classes found'
        
        if cpt > arguments.number_of_classes:
            break
    
    
    classes[0][arguments.number_of_classes-1] = classes[0][255]
    
    for i in range(0, 256):
        if classes[0][i][0] == -1 and classes[0][i][1] == -1 and classes[0][i][2] == -1:
            classes[0][i] = [0,0,0]
    classes[0][255] = [0,0,0]
    
    
    print classes
    img_new = Image.new('RGB', (len(classes[0]), len(classes)))
    img_new.putdata([tuple(p) for row in classes for p in row])
    img_new.save(arguments.output)




if __name__ == '__main__':
    
    main(None)
    
    pass


