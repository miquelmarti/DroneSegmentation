#!/usr/bin/env python

from PIL import Image


class FileListIterator(object):
    listFile = None

    def __init__(self, listFileName, pairs=False, sep=' '):
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
        if self.pairs:
            nextLabelImg = Image.open(p[2].strip())
        return (nextImg, nextLabelImg)

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
            return (Image.fromarray(frame, 'RGB'), None) # no labels for videos
        else:
            raise StopIteration()

    def __del__(self):
        if type(self.videoCapture) is not None:
            self.videoCapture.release()
