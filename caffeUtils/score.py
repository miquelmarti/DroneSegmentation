# Based on Evan Shelhamer's score.py from fcn.berkeleyvision.org

from __future__ import division
import numpy as np
import caffe
from PIL import Image
import logging
logger = logging.getLogger()


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
def preProcessing(image, mean=None):
    # Get pixel values and convert them from RGB to BGR
    image = np.array(image, dtype=np.float32)
    if len(image.shape) is 2:
        image = np.resize(image, (image.shape[0], image.shape[1], 3))
    image = image[:, :, ::-1]

    # Substract mean pixel values of pascal training set
    if mean is not None:
        image -= mean

    # Reorder multi-channel image matrix from W x H x C to C x H x W expected
    # by Caffe
    image = image.transpose((2, 0, 1))

    return image


def fast_hist(a, b, n):
    try:
        k = (a >= 0) & (a < n)
        bincount = np.bincount(n * a[k].astype(int) + b[k], minlength=n**2)
        # Element (i,j) of the returned array is the number of pixels with label
        # i in a and label j in b.  So the diagonal element (i,i) is the number of
        # pixels in class i that were correctly labelled.
        output = bincount.reshape(n, n)
    except: #Sometimes, bincount is not reshapeble by n,n
        k = (a >= 0) & (a < n)
        bincount = np.bincount(n * a[k].astype(int) + b[k]*(n * a[k].astype(int) + b[k]<n**2), minlength=n**2)
        # Element (i,j) of the returned array is the number of pixels with label
        # i in a and label j in b.  So the diagonal element (i,i) is the number of
        # pixels in class i that were correctly labelled.
        output = bincount.reshape(n, n)
    return output


# TODO: Deal with classification (here, only pixel-wise)
# TODO: What about those without the labelLayer ?
def runNetForward(net, image=None, gtLabels=None, dataLayer='data',
                  lossLayer='loss', outLayer='score', labelLayer='label'):
    if image is not None:
        net.blobs[dataLayer].reshape(1, *image.shape)
        net.blobs[dataLayer].data[...] = image
    
    net.forward()
    output = net.blobs[outLayer].data
    
    if output.shape and (gtLabels is not None or labelLayer in net.blobs):
        flatOutput = output[0].argmax(0).flatten()
        hist = None
        n_cl = net.blobs[outLayer].channels
        if gtLabels is not None:
            hist = fast_hist(np.array(gtLabels).flatten(), flatOutput, n_cl)
        else:
            hist = fast_hist(net.blobs[labelLayer].data[0, 0].flatten(),
                             flatOutput, n_cl)
        return output, hist, net.blobs[lossLayer].data.flat[0]
    else:
        # TODO: if labelLayer in net.blobs and not output.shape
        # TODO: -> classification, take the output (printable) in charge
        return output, None, 0


def segmentImage(net, image, lossLayer, outLayer, mean=None, gtLabels=None):
    image = preProcessing(image, mean)
    return runNetForward(net, image, gtLabels=gtLabels,
                         lossLayer=lossLayer, outLayer=outLayer,
                         labelLayer=None)


def computeSegmentationScores(hist):
    # Often happens in classification (instead of semantic segmentation)
    if hist is None:
        return SegScores(None, 0, 0, 0, 0, 0)
    
    # overall accuracy
    ovAcc = np.diag(hist).sum() / hist.sum()
    # per-class accuracy
    meanAcc = np.nanmean(np.diag(hist) / hist.sum(1))
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    meanIu = np.nanmean(iu)
    freq = hist.sum(1) / hist.sum()
    # weighted average of the IU (better in case of imbalanced classes)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return SegScores(None, ovAcc, meanAcc, iu, meanIu, fwavacc)


def scoreDataset(deployFilename, modelFilename, dataset, mean=None,
                 lossLayer='loss', outLayer='score'):
    net = caffe.Net(str(deployFilename), str(modelFilename), caffe.TEST)
    hlZip = [segmentImage(net, image, lossLayer, outLayer, mean, gtLabels)[1:]
             for image, gtLabels, _ in dataset]
    hists, losses = zip(*hlZip)
    hist = sum(hists)
    scores = computeSegmentationScores(hist)
    scores.loss = np.mean(losses)
    return scores
