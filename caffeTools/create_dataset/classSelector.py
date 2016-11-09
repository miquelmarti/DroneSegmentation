
from PIL import Image
import numpy as np
import scipy.io
import cv2

listFileName = '/home/shared/datasets/SBD_dataset/dataset/train.txt'
newFileName = '/home/shared/datasets/SBD_dataset/dataset/train_badClasses.txt'
lookingForClass = np.array([2, 9, 11, 18])

listImages = []



listFile = open(listFileName, 'r')


for line in listFile:
        p = line.partition(' ')
        imagePath = p[0].strip()
        
        print "Looking in", imagePath
        
        #Open nextLabelImg
        mat = scipy.io.loadmat('/home/shared/datasets/SBD_dataset/dataset/cls/{}.mat'.format(imagePath))
        labelImg = mat['GTcls'][0]['Segmentation'][0].astype(np.uint8)
        
        image = cv2.imread('/home/shared/datasets/SBD_dataset/dataset/img/{}.jpg'.format(imagePath))
        
        #Look if the classes we want are inside
        keepImage = False
        for currentClass in lookingForClass:
                if np.sum(labelImg==currentClass) > 0:
                        keepImage = True
                        
        # If yes, store the image path
        if keepImage:
                listImages.append(imagePath)
                print "Classes found"
                cv2.imshow("Image", image)
                cv2.waitKey(0)
        else:
                print "Classes not found"
listFile.close()                

##Prints the image list in a new file
#
#newFile = open(newFileName, 'w')
#for newImagePath in listImages:
#        newFile.write(newImagePath+'\n')
#newFile.close()
