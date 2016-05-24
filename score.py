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
    # Element (i,j) of the returned array is the number of pixels with label
    # i in a and label j in b.  So the diagonal element (i,i) is the number of
    # pixels in class i that were correctly labelled.
    return bincount.reshape(n, n)


def runNetForward(net, image=None, gtImage=None, dataLayer=DATA_LAYER_NAME,
                  lossLayer='loss', outLayer='score', labelLayer='label'):
    if image is not None:
        net.blobs[dataLayer].reshape(1, *image.shape)
        net.blobs[dataLayer].data[...] = image
    net.forward()
    output = net.blobs[outLayer].data
    if gtImage is not None or labelLayer in net.blobs:
        flatOutput = output[0].argmax(0).flatten()
        hist = None
        n_cl = net.blobs[outLayer].channels
        if gtImage is not None:
            hist = fast_hist(np.array(gtImage).flatten(), flatOutput, n_cl)
        else:
            hist = fast_hist(net.blobs[labelLayer].data[0, 0].flatten(),
                             flatOutput, n_cl)
        return output, hist, net.blobs[lossLayer].data.flat[0]
    else:
        return output


def segmentImage(net, image, lossLayer, outLayer, mean=None,
                 groundTruthImage=None, newShape=None):
    image = preProcessing(image, mean, newShape)
    return runNetForward(net, image, gtImage=groundTruthImage,
                         lossLayer=lossLayer, outLayer=outLayer,
                         labelLayer=None)


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
                 lossLayer='loss', outLayer='score'):
    net = caffe.Net(str(deployFilename), str(modelFilename), caffe.TEST)
    hlZip = [segmentImage(net, image, lossLayer, outLayer, mean,
                          groundTruthImage)[1:]
             for image, groundTruthImage in dataset]
    hists, losses = zip(*hlZip)
    hist = sum(hists)
    scores = computeSegmentationScores(hist)
    scores.loss = np.mean(losses)
    return scores
