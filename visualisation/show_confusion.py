import numpy as np
import cv2
from matplotlib import mpl,pyplot

#The hist array is composed this way : hist[:,C] will show the real groundtruths that the model guessed being class C. hist[C,:] will show the model's guessed output when the real groundtruth was C
# hist[i,j] is the number of pixels groundtruth said class i when model said class j

hist = np.load('/home/johannes/hist.npy')
hist = hist / hist.sum(axis=1) #Normalisation by the number of pixels in groundtruth

for i in range(0,hist.shape[0]):
        hist[i,i] = 0*hist[i,i] #Suppresses diagonal (because diagonal >> other values, messing up the display)


cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',['blue','green', 'orange','red'],256)
img2 = pyplot.imshow(hist,interpolation='nearest',cmap = cmap2,origin='lower')

pyplot.colorbar(img2,cmap=cmap2)
pyplot.ylabel("Groundtruth class ")
pyplot.xlabel("Class output by model")
pyplot.show()

