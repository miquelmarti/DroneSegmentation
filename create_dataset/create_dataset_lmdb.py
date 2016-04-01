# Create the dataset with images in the file_with_paths
# python create_dataset_lmdb.py

import caffe
import lmdb
from PIL import Image
import numpy as np
import glob
from random import shuffle


H = 500 # Required Height
W = 500 # Required Width
dataset_name = '/home/pierre/pascal_2012/train_lmdb/' # Folder where to create the dataset
file_with_paths = '/home/shared/datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt' # Text file with all the paths
labels = True # Set to true if ground truth
number_of_classes = 21 # Number of classes in the dataset




def main(args):
    
    in_db = lmdb.open(dataset_name, map_size=int(1e12))
    with in_db.begin(write=True) as in_txn:
        for in_idx, in_ in enumerate(open(file_with_paths)):
            print 'img: ' + str(in_idx)
            
            # load image:
            im = np.array(Image.open(in_.rstrip()))
            # save type
            Dtype = im.dtype
            
            if labels:
                # Resize the input image
                Limg = Image.fromarray(im)
                Limg = Limg.resize([H, W],Image.NEAREST)
                im = np.array(Limg,Dtype)
                # Convert from HxWxC (C=3) to HxWxC (C=1)
                im = im.reshape(im.shape[0],im.shape[1],1)
            else:
                # RGB to BGR
                im = im[:,:,::-1]
                im = Image.fromarray(im)
                im = im.resize([H, W], Image.ANTIALIAS)
                im = np.array(im,Dtype)
            
            # Convert to CxHxW
            im = im.transpose((2,0,1))
            if labels:
                im[im==255]=number_of_classes
            
            # Create the dataset
            im_dat = caffe.io.array_to_datum(im)
            in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
    in_db.close()




if __name__ == '__main__':
    
    main(None)
    
    pass



