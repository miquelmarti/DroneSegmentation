#!/usr/bin/env python

import cv2
import numpy as np
import argparse
import os

OUT_PREFIX = 'boundary_'
CV2_LOAD_IMAGE_UNCHANGED = -1
VOC_BOUNDARY_PX_VAL = 21
BOUNDARY_PX_VAL = 1


def borderifyPixel(inImg, outImg, targetPixel, comparePixel):
    targetPxVal = inImg[targetPixel]
    comparePxVal = inImg[comparePixel]
    if np.any(targetPxVal != comparePxVal):
        outImg[comparePixel] = BOUNDARY_PX_VAL
        outImg[targetPixel] = BOUNDARY_PX_VAL


def processImage(inputFilename, outputFilename, hasBounds=False):
    inImg = cv2.imread(inputFilename, CV2_LOAD_IMAGE_UNCHANGED)
    if inImg is None:  # this can happen if file is not an imagePath
        print 'WARNING: non-image file', inputFilename, 'encountered'
        return
    if len(inImg.shape) > 2:
        inImg = inImg[:, :, 0]
    outImg = np.zeros(inImg.shape)
    if hasBounds:
        outImg[inImg == VOC_BOUNDARY_PX_VAL] = BOUNDARY_PX_VAL
    else:
        # iterate over the imagePath pixel-by-pixel (skip final row and col)
        for i in range(inImg.shape[0] - 1):
            for j in range(inImg.shape[1] - 1):
                pixel = (i, j)
                borderifyPixel(inImg, outImg, pixel, (i+1, j))
                borderifyPixel(inImg, outImg, pixel, (i, j+1))

    cv2.imwrite(outputFilename, outImg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--image', help='a ground-truth imagePath to convert.')
    group.add_argument('--images', help='\
    a directory containing ground-truth imagePaths to convert.')
    
    parser.add_argument('--bounds', action="store_true", help='\
    treat the image as if boundaries already present.')

    parser.add_argument('out_location', help='\
    The output location of the result.  If --imagePath is specified, this is \
    treated as a filename.  If --imagePaths is specified, this must be an \
    existing directory.')

    # TODO add support for lists of files.
    # TODO allow user to specify out-dir.
    args = parser.parse_args()

    images = []
    if args.image:
        images.append(args.image)
    elif args.images:
        images = os.listdir(args.images)
    else:
        print "ERROR: No output location specified. Run with '-h' for details."
        exit(1)
        
    for image in images:
        print 'converting', image
        outPath = os.path.join(args.out_location, image)
        imagePath = os.path.join(args.images, image)
        processImage(imagePath, outPath, args.bounds)
