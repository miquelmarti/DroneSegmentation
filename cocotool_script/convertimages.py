from pycocotools.coco import COCO
import numpy as np
import cv2
import skimage.io as io
import pylab as pl
import mask
import os
import argparse

def get_arguments():
    # Import arguments
    parser = argparse.ArgumentParser()
    # Mandatory arguments
    parser.add_argument('--dst', type=str, help='\
    destination folder to save png files')
    parser.add_argument('--data', type=str, default='/home/shared/datasets/MS_COCO', help='\
    path of images folder where jpg files can be found')
    
    return parser.parse_args()

def drawSegmentation(image, anns, img):
        """
        draws segmentation on input image
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        if 'segmentation' in anns[0]:
	    # sort annotations from biggest to smallest to avoid occlusions
	    anns.sort(key=lambda x: x['area'], reverse=True)
            for ann in anns:
		# open file making the conversation MSCOCO classes -> VOC classes
		f = open('classes.txt', 'r')
		for line in f:
			splt = line.split('\t')
			if ann['category_id'] == int(splt[0]):
				pixelvalue = int(splt[1])
				break
		f.close()
		c = [pixelvalue, pixelvalue, pixelvalue]

		if type(ann['segmentation']) == list:
                    # polygon
                    for seg in ann['segmentation']:
			poly = np.array(seg).reshape((len(seg)/2, 2))
			pts = np.array(poly, np.int32)
			pts.reshape((-1,1,2))
			cv2.polylines(image,[pts],True,(255,255,255), 3)			
			cv2.fillPoly(image, [pts], c)
		else:
                    # mask

		    t = coco.imgs[ann['image_id']]
                    if type(ann['segmentation']['counts']) == list:
                        rle = mask.frPyObjects([ann['segmentation']], t['height'], t['width'])
                    else:
                        rle = [ann['segmentation']]
                    m = mask.decode(rle)
                    img = np.ones( (m.shape[0], m.shape[1], 3) )
                    for i in range(3):
                        img[:,:,i] = pixelvalue
		    mask2 = np.dstack( (img, m) )
		    for x in range(img.shape[0]):
			for y in range(img.shape[1]):
				if not mask2[x][y][3] == 0:
					image[x][y] = c
		    
		 


if __name__ == '__main__':

	# Get all options
	args = get_arguments()
	

	dataDir=args.data
	dst=args.dst	
	dataType='train2014'#val2014
	annFile='%s/annotations/instances_%s.json'%(dataDir,dataType)
	# initialize COCO api for instance annotations
	coco=COCO(annFile)

	#get VOC category ids
	catIds = coco.getCatIds(['person','bicycle','car','motorcycle','airplane','bus','train','boat','bird','cat','dog','horse','sheep','cow','bottle','chair','couch','potted plant','dining table','tv'])

	#get all image ids
	imgIds = coco.getImgIds()

	#load all images
	imgs = coco.loadImgs(imgIds)

	imgs.sort(key=lambda x: x['id'])

	for img in imgs:
		#open image
		image = cv2.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']),-1)
		#if there is no color channels, resize the matrix	
		if len(image.shape) is 2:
		        image = np.resize(image, (image.shape[0],image.shape[1],3))
		#set all pixel to background class
	 	image[:] = 0
		# load instance annotations
		annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=True) + coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=False)
		anns = coco.loadAnns(annIds)
	
		# if there is no annotation loaded, it means no VOC class is present in the picture
		if len(anns)==0:
			continue

		#apply segmentations on image	
		drawSegmentation(image, anns, img)
	
		print img['file_name']
		
		#save files
		if not os.path.isdir(dst):
			os.makedirs(dst)
		path=dst+'/'+str(img['file_name'])[:-4]+'.png'

		cv2.imwrite(path, image)


