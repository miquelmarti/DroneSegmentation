#!/usr/bin/env python

# Get the weights per classes of a dataset (following this paper : http://arxiv.org/pdf/1411.4734v4.pdf)
# Ex : python /home/pierre/hgRepos/caffeTools/get_class_weights.py /home/shared/datasets/CamVid/train_lab.txt 12


import numpy as np
import argparse
from PIL import Image
import cv2


# Class for reading the file listing the labels
class FileListIterator(object):
    listFile = None

    def __init__(self, listFileName, sep=' '):
        self.listFile = open(listFileName, 'r')
        self.sep = sep
    
    def __iter__(self):
        return self

    def next(self):
        # Image from PIL for getting the number of the classes (and not the colours with cv2)
        nextLine = self.listFile.next()
        nextImg = Image.open(nextLine.rstrip())
        return nextImg

    def __del__(self):
        if type(self.listFile) is file:
            self.listFile.close()


def get_arguments():
    # Import arguments
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    
    # Mandatory options
    parser.add_argument('path_to_txt', type=str, \
                                    help=   'Path to the txt file with the name of all labels')
    parser.add_argument('nb_class', type=int, \
                                    help=   'Number of classes in the dataset')
    
    return parser.parse_args()


def median(lst):
    sortedLst = sorted(lst)
    lstLen = len(lst)
    index = (lstLen - 1) // 2

    if (lstLen % 2):
        return sortedLst[index]
    else:
        return (sortedLst[index] + sortedLst[index + 1])/2.0
        


if __name__ == '__main__':

    # Get all options
    args = get_arguments()

    # Create the appropriate iterator
    imageIterator = FileListIterator(args.path_to_txt)
    
    # Array with all the pixels
    array = []
    
    # Main loop, for each image to process
    for real_label in imageIterator:
        
        # Convert image
        rl = np.array(real_label)
        
        # If pascal VOC, reshape the label to HxWx1s
        if len(rl.shape) == 3:
            rl = rl[:,:,0]
        
        # Get the number of pixels per class for this image
        tmp = [0]*args.nb_class
        for i in range(0, len(rl)):
            for j in range(0, len(rl[i])):
                # the class args.nb_class is the border, we don't want it
                if rl[i][j] != args.nb_class:
                    tmp[rl[i][j]] += 1
        array.append(tmp)
        
        print 'img ', len(array), ' -> done'
    
    
    # Get freq(c)
    freq_c = [0]*args.nb_class
    for i in range(0, args.nb_class):
        tmpSum = 0
        freq_c[i] = sum(np.array(array)[:,i])
        for j in range(0, len(array)):
            if array[j][i] != 0:
                tmpSum += sum(array[j])
        if tmpSum != 0:
            freq_c[i] = float(freq_c[i]) / tmpSum
    
    # Compute median
    median = median(freq_c)
    
    # Compute freq
    freq = [0]*len(freq_c)
    for i in range(0, len(freq)):
        if freq_c[i] != 0:
            freq[i] = float(median) / freq_c[i]
    
    # Display the class_weighting
    print '  loss_param: {'
    print '    weight_by_label_freqs: true'
    print '    ignore_label: ', args.nb_class
    for i in range(0, args.nb_class):
        print '    class_weighting: ', round(freq[i], 4)
    print '  }'


