from pycocotools.coco import COCO
import numpy as np
import cv2
import os
import argparse
import json
import itertools

def get_arguments():
    # Import arguments
    parser = argparse.ArgumentParser()
    # Mandatory arguments
    parser.add_argument('--dst', type=str, help='\
    destination folder to save png files')
    parser.add_argument('--data', type=str, help='\
    path of images folder where jpg files can be found')
    parser.add_argument('--ann', type=str, help='annotation file path', default='')
    parser.add_argument('--color', type=bool, help='input color images', default=False)
    return parser.parse_args()

def getPriority(id):
	#if annotation is road, footpath, grass or earth
	# swiss 100 
	#if id == 3 or id == 4 or id == 6 or id == 7:
	#okutama
	if id == 5 or id == 4 or id == 3 or id == 13 or id == 14:
		return 1
	return 0

def drawSegmentation(image, anns, color):
        """
        draws segmentation on input image
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return False
        if 'segmentation' in anns[0]:
	    # sort annotations from biggest to smallest to avoid occlusions
	    colours = cv2.imread('/home/shared/data/datasets/Okutama/colours/okutama_colours.png').astype(np.uint8)
	    anns.sort(key=lambda x: (getPriority(x['category_id']),x['area']), reverse=True)
            for ann in anns:
		#print human readable colors
		if color:
			c = [int(colours[0][ann['category_id']][0]), int(colours[0][ann['category_id']][1]), int(colours[0][ann['category_id']][2])] 
		else:
			c = [ann['category_id'], ann['category_id'], ann['category_id']]
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
	cats = coco.loadCats(catIds)

	#get all image ids
	imgIds = coco.getImgIds()
	
	#load all images
	imgs = coco.loadImgs(imgIds)

	#sort images by id
	imgs.sort(key=lambda x: x['id'])

	cpt = 1

	for img in imgs:
		#open image
		#image = cv2.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']),-1)
		for subdir, dirs, files in os.walk(dataDir):
			if not os.path.isfile(subdir+"/"+img['file_name']):
				continue
			image = cv2.imread('%s/%s'%(subdir,img['file_name']),-1)
			break
				
		
		#if there is no color channels, resize the matrix
		
		if len(image.shape) is 2:
		        image = np.resize(image, (image.shape[0],image.shape[1],3))
		
		#set all pixels to background class
	 	image[:] = 0
		# load instance annotations

		lists = [coco.imgToAnns[img['id']] for imgId in imgIds if imgId in coco.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))

		#annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds)
		#anns = coco.loadAnns(annIds)
		
		#apply segmentations on image	
		suitable = drawSegmentation(image, anns, args.color)
		
		if not suitable:
			continue

		print img['file_name'], "("+ str(cpt) + "/"+ str(len(imgs)) + ")"
		cpt += 1
		#display image

		#cv2.imshow('image',image)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()

		#save image

		if not os.path.isdir(dst):
			os.makedirs(dst)
		path=dst+'/'+str(img['file_name'])[:-4]+'.png'
		cv2.imwrite(path, image)




