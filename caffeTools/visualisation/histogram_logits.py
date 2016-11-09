import matplotlib.pyplot as plt
import numpy as np


#listFileName = '/home/shared/caffeDeepLab/exper/voc12/features2/deeplab_largeFOV/val/fc8/list.txt'
listFileName = '/home/johannes/NetOutputs/Swiss50/list_todelete.txt'
maxiter = 1



listFile = open(listFileName, 'r')
matrix = np.array(())

iter_ = 0
for line in listFile:
        p = line.partition(' ')
        a = np.load(p[2].strip()).squeeze()
	a = a[:,100,100]
        #a = a.reshape((a.shape[0]*a.shape[2]*a.shape[1])).squeeze()
        matrix = np.insert(a,0,matrix)
        print p[2].strip()
        iter_ = iter_ + 1
        if iter_ >= maxiter:
                break

plt.hist(matrix, bins=100)
plt.show()
