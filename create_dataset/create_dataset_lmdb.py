# Create the dataset with images in the file_with_paths
# python create_dataset_lmdb.py --W 480 --H 360 --output_path /home/pierre/camvid/val_gt_lmdb/ --input_images /home/shared/datasets/CamVid/val_lab.txt --nb_classes 13 --labels True

import caffe
import lmdb
from PIL import Image
import numpy as np
import glob
from random import shuffle
import argparse

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
    parser.add_argument('--nb_classes', type=str, default=255, \
                                    help=   'Number of classes. Only required for labels')
    parser.add_argument('--labels', type=bool, default=False, \
                                    help=   'Precise if dealing with labels and not RGB')
    return parser.parse_args()

def main(args):
    
    # Get all options
    args = get_arguments()
    
    in_db = lmdb.open(args.output_path, map_size=int(1e12))
    with in_db.begin(write=True) as in_txn:
        for in_idx, in_ in enumerate(open(args.input_images)):
            print 'Loading image ', str(in_idx), ' : ', in_.rstrip()
            
            # load image:
            im = np.array(Image.open(in_.rstrip()))
            # save type
            Dtype = im.dtype
            
            if args.labels:
                # Resize the input image
                Limg = Image.fromarray(im)
                Limg = Limg.resize([args.W, args.H],Image.NEAREST)
                im = np.array(Limg,Dtype)
                # Convert from HxWxC (C=3) to HxWxC (C=1)
                #for i in range(0, im.shape[0]):
                #    for j in range(0, im.shape[1]):
                #        if im[i][j][0] != 0:
                #            print im[i][j]
                im = im[:,:,0].reshape(im.shape[0],im.shape[1],1)
            else:
                # RGB to BGR
                im = im[:,:,::-1]
                im = Image.fromarray(im)
                im = im.resize([args.W, args.H], Image.ANTIALIAS)
                im = np.array(im,Dtype)
            
            # Convert to CxHxW
            im = im.transpose((2,0,1))
            if args.labels:
                im[im==255]=args.nb_classes
            
            # Create the dataset
            im_dat = caffe.io.array_to_datum(im)
            in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
    in_db.close()




if __name__ == '__main__':
    
    main(None)
    
    pass



