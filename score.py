# Based on Evan Shelhamer's score.py from fcn.berkeleyvision.org

from __future__ import division
import numpy as np
import caffe
from PIL import Image

DATA_LAYER_NAME = "data"


class SegScores(object):
    """A simple struct to contain the segmentation scores we compute."""
    def __init__(self, loss, ovAcc, meanAcc, iu, meanIu, fwavacc):
        self.loss = loss
        self.overallAcc = ovAcc
        self.meanAcc = meanAcc
        self.iu = iu
        self.meanIu = meanIu
        self.fwavacc = fwavacc


# def preProcessing(img, shape, resize_img, mean=None):
def preProcessing(image, mean=None, newShape=None):
    # Ensure that the image has the good size
    # try without resizing for now
    if newShape is not None:
        image = image.resize((newShape[3], newShape[2]), Image.ANTIALIAS)

    # Get pixel values and convert them from RGB to BGR
    image = np.array(image, dtype=np.float32)
    image = image[:, :, ::-1]

    # Substract mean pixel values of pascal training set
    if mean is not None:
        image -= mean

    # Reorder multi-channel image matrix from W x H x C to C x H x W expected
    # by Caffe
    image = image.transpose((2, 0, 1))

    return image


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    bincount = np.bincount(n * a[k].astype(int) + b[k], minlength=n**2)
    return bincount.reshape(n, n)


def segmentImage(net, image, inLayer, outLayer, mean=None,
                 groundTruthImage=None, newShape=None):
    image = preProcessing(image, mean, newShape)
    net.blobs[DATA_LAYER_NAME].reshape(1, *image.shape)
    net.blobs[DATA_LAYER_NAME].data[...] = image
    net.forward()
    if groundTruthImage is not None:
        n_cl = net.blobs[outLayer].channels
        hist = fast_hist(np.array(groundTruthImage).flatten(),
                         net.blobs[outLayer].data[0].argmax(0).flatten(),
                         n_cl)
        loss = net.blobs[inLayer].data.flat[0]
        return net.blobs[outLayer].data, hist, loss
    else:
        return net.blobs[outLayer].data


def computeSegmentationScores(hist):
    # overall accuracy
    ovAcc = np.diag(hist).sum() / hist.sum()
    # per-class accuracy
    meanAcc = np.nanmean(np.diag(hist) / hist.sum(1))
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    meanIu = np.nanmean(iu)
    freq = hist.sum(1) / hist.sum()
    # not sure what this is...Shelhamer computes it, so we retain it.
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return SegScores(None, ovAcc, meanAcc, iu, meanIu, fwavacc)


def scoreDataset(deployFilename, modelFilename, dataset, mean=None,
                 inLayer='loss', outLayer='score'):
    
    net = caffe.Net(str(deployFilename), str(modelFilename), caffe.TEST)
    print 'Segment the validation set...'
    hlZip = [segmentImage(net, image, inLayer, outLayer, mean,
                          groundTruthImage)[1:]
             for image, groundTruthImage in dataset]
    dataset.reset()
    hists, losses = zip(*hlZip)
    hist = sum(hists)
    scores = computeSegmentationScores(hist)
    scores.loss = np.mean(losses)
    return scores
