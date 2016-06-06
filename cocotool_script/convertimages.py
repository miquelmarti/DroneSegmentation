from pycocotools.coco import COCO
import numpy as np
import cv2
import os
import argparse
import json

def get_arguments():
    # Import arguments
    parser = argparse.ArgumentParser()
    # Mandatory arguments
    parser.add_argument('--dst', type=str, help='\
    destination folder to save png files')
    parser.add_argument('--data', type=str, help='\
    path of images folder where jpg files can be found')
    parser.add_argument('--ann', type=str, help='annotation file path', default='')
    return parser.parse_args()

def drawSegmentation(image, anns, img):
        """
        draws segmentation on input image
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return False
        if 'segmentation' in anns[0]:
	    # sort annotations from biggest to smallest to avoid occlusions
	   
	    anns.sort(key=lambda x: x['area'], reverse=True)
            for ann in anns:
		
		pixelvalue = ann['category_id']* 10
		c = [pixelvalue, pixelvalue, pixelvalue] 
		if type(ann['segmentation']) == list:
		    poly = np.array(ann['segmentation']).reshape((len(ann['segmentation'])/2, 2))
		    pts = np.array(poly, np.int32)
		    pts.reshape((-1,1,2))
		    #cv2.polylines(image,[pts],True,(255,255,255), 3)			
		    cv2.fillPoly(image, [pts], c)
	return True
		    
		 


if __name__ == '__main__':

	# Get all options
	args = get_arguments()
	

	dataDir=args.data
	dst=args.dst	
	annFile=args.ann

	# initialize COCO api for instance annotations
	coco=COCO(annFile)

	#get all category ids
	catIds = coco.getCatIds()

	#get all image ids
	imgIds = coco.getImgIds()
	
	#load all images
	imgs = coco.loadImgs(imgIds)

	#sort images by id
	imgs.sort(key=lambda x: x['id'])

	for img in imgs:
		#open image
		#image = cv2.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']),-1)
		image = cv2.imread('%s/%s'%(dataDir,img['file_name']),-1)		
		#if there is no color channels, resize the matrix
		
		if len(image.shape) is 2:
		        image = np.resize(image, (image.shape[0],image.shape[1],3))
		
		#set all pixel to background class
	 	image[:] = 0
		# load instance annotations
		annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds)		
		anns = coco.loadAnns(annIds)

		#apply segmentations on image	
		suitable = drawSegmentation(image, anns, img)
		
		if not suitable:
			continue

		print img['file_name']

		#display image
		#cv2.imshow('image',image)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()

		#save files
		if not os.path.isdir(dst):
			os.makedirs(dst)
		path=dst+'/'+str(img['file_name'])[:-4]+'.png'

		cv2.imwrite(path, image)




