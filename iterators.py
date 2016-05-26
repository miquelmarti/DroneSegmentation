#!/usr/bin/env python

from PIL import Image
import cv2


class FileListIterator(object):
    listFile = None

    def __init__(self, listFileName, pairs=False, sep=' ', backup=None):
        self.listfileName = listFileName
        self.listFile = open(listFileName, 'r')
        self.pairs = pairs
        self.sep = sep
    
    def __iter__(self):
        return self

    def next(self):
        nextLine = self.listFile.next()
        p = nextLine.partition(self.sep)
        nextImg = Image.open(p[0].strip())
        nextLabelImg = None
        imagePath = p[0].strip()
        if self.pairs:
            nextLabelImg = Image.open(p[2].strip())
        return (nextImg, nextLabelImg, imagePath)

    def reset(self):
        self.__init__(self.listfileName, self.pairs, self.sep)

    def __del__(self):
        if type(self.listFile) is file:
            self.listFile.close()


class VideoIterator(object):
    videoCapture = None

    def __init__(self, videoFileName):
        self.videoCapture = cv2.VideoCapture(videoFileName)
    
    def __iter__(self):
        return self

    def next(self):
        rval, frame = self.videoCapture.read()
        if rval:
            # no labels for videos
            return (Image.fromarray(frame, 'RGB'), None, None)
        else:
            raise StopIteration()

    def __del__(self):
        if type(self.videoCapture) is not None:
            self.videoCapture.release()
