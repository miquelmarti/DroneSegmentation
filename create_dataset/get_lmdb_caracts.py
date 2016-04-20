# Get few caracteristics of lmdb_folder
# python get_lmdb_caracts.py /home/shared/datasets/VOCdevkit/VOC2012/LMDB

# TODO : Add an analysis of the caracteristics


import caffe
import lmdb
import argparse
import os
import caffe.proto.caffe_pb2
from caffe.io import datum_to_array


def get_arguments():
    # Import arguments
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    
    # Mandatory options
    parser.add_argument('lmdb_folder', type=str, \
                                    help=   'Path to the LMDB folder model (usually [...]/deploy.prototxt)')
    
    return parser.parse_args()
        


if __name__ == '__main__':
    
    # LMDB datasets name
    names = ['train_lmdb', 'train_gt_lmdb', 'val_lmdb', 'val_gt_lmdb']
    
    # Shapes and nb of images in dataset
    shapes = []
    nbOfImages = []
    min_max = []

    # Get all options
    args = get_arguments()
    
    # For each datasets (usually train, train_gt, val and val_gt)
    for n in range(0, len(names)):
        
        # Open the LMDB folder
        lmdb_env = lmdb.open(os.path.join(args.lmdb_folder, names[n]))
        lmdb_txn = lmdb_env.begin()
        lmdb_cursor = lmdb_txn.cursor()
        datum = caffe.proto.caffe_pb2.Datum()
        
        # Init values for new dataset
        shapes.append([])
        nbOfImages.append(0)
        
        # Get the min and max of the images (to know if it processes it well)
        _min = _max = []
        
        # Read LMDB images per images
        for key, value in lmdb_cursor:
            nbOfImages[n] += 1
            datum.ParseFromString(value)
            label = datum.label
            data = caffe.io.datum_to_array(datum)
            if (data.shape in shapes[n]) == False:
                shapes[n].append(data.shape)
            
            _min.append(min([d.min() for d in data]))
            _max.append(max([d.max() for d in data]))
            
        min_max.append([min(_min), max(_max)])
    
    
    # Display results
    caract = ''
    for n in range(0, len(names)):
        caract += 'Dataset :\t\t' + str(names[n]) + '\n'
        caract += 'Contains :\t\t' + str(nbOfImages[n]) + ' images\n'
        caract += 'With the shape(s) :\t' + str(shapes[n]) + '\n'
        caract += 'All these images have, as values, numbers between ' + str(min_max[n][0]) + ' and ' + str(min_max[n][1]) + '\n\n'
    
    # Save the caracteristics
    f = open(os.path.join(args.lmdb_folder, 'lmdb_caracteristics.md'), 'w')
    f.write(caract)
    f.close()
    
    # Also display them
    print caract





