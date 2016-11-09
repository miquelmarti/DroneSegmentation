import caffe

import numpy as np
from PIL import Image, ImageEnhance
import cv2

import random
import os


class SemSegDataLayer(caffe.Layer):
    """
    This class serves as a template class, and is not to be used directly.
    Use an appropriate subclass instead.
    """
    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - dataset_dir: path to dataset's top-level directory
        - image_list: path from dataset_dir to the image-index text file
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)
        """
        # config
        params = eval(self.param_str)
        image_list = params['image_list']
        self.dataset_dir = params.get('dataset_dir', None)
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', False)
        self.seed = params.get('seed', None)
        
        # data augmentation
        mirror, jitter, noise, crop = False, 0, False, (0,0)
        if 'mirror' in params:
            mirror = params['mirror']
        if 'jitter' in params:
            jitter = params['jitter']
        if 'noise' in params:
            noise = params['noise']
        if 'crop' in params:
            crop = params['crop']
        self.mirror = mirror
        self.jitter = jitter
        self.noise = noise
        self.crop = crop
        self.isLabel = False
        self.mirror_param = (False, '')
        self.crop_param = (False, (0,0))

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        if self.dataset_dir is not None:
            image_list = os.path.join(self.dataset_dir, image_list)
        with open(image_list, 'r') as f:
            self.samples = f.read().splitlines()
        self.idx = 0
            
        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.samples)-1)
        else:
            self.idx = 0

    def reshape(self, bottom, top):
        # load image + label image pair
        in_ = self.load_image(self.samples[self.idx])
        # cast to float
        in_ = np.array(in_, dtype=np.float32)
        if len(in_.shape) == 2:
            # add a third dimension to grayscale images
            in_ -= np.mean(self.mean)
            in_ = in_[:, :, np.newaxis].repeat(repeats=3, axis=2)
        else:
            # switch channels RGB -> BGR
            in_ = in_[:, :, ::-1]
            in_ -= self.mean
        # subtract mean
        # transpose to channel x height x width order
        in_ = in_.transpose((2, 0, 1))
        self.data = in_
        
        self.label = self.load_label(self.samples[self.idx])
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)
        
    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.samples)-1)
        else:
            self.idx += 1
            if self.idx == len(self.samples):
                self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

    # You must define these when subclassing this class:
    # def load_image(self, fileLine):
    #     pass

    # def load_label(self, fileLine):
    #     pass


class ImgPairFileDataLayer(SemSegDataLayer):

    def load_image(self, fileLine):
        filename = fileLine.split()[0]
        img = self.load_image_helper(filename)
        self.isLabel = True
        return img
        
    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        filename = idx.split()[1].strip()
        im = self.load_image_helper(filename)
        label = np.array(im, dtype=np.uint8)
        if len(label.shape) > 2:
            label = label[:, :, 0]
        label = label[np.newaxis, ...]
        
        self.isLabel = False
        return label
        
    def load_image_helper(self, filename):
        if self.dataset_dir is not None:
            # treat the filename as relative
            if os.path.isabs(filename):
                filename = filename[1:]
            filename = os.path.join(self.dataset_dir, filename)
        image = cv2.imread(filename)
        if not self.isLabel:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Data augmentation
        if self.isLabel:
            if self.mirror_param[0]:
                image = self.do_flip(image, self.mirror_param[1])
                self.mirror_param = (False, '')
            if self.crop_param[0]:
                image = self.do_random_crop(image, self.crop_param[1])
                self.crop_param = (False, (0,0))
        else:
            if self.mirror:
                image = self.do_flip(image)
            if self.jitter != 0:
                image = self.do_image_jittering(image)
            if self.noise:
                image = self.apply_noise(image)
            if self.crop != (0,0):
                image = self.do_random_crop(image)
        
        return Image.fromarray(image)
        #return Image.open(filename)
        
    def do_flip(self, image, sense=''):
        if sense != '':
            image = image[:, ::-1, :] if sense == 'h' else image[::-1, :, :]
            return image
        
        if bool(np.random.choice(2)):
            if bool(np.random.choice(2)): # Horizontal flip
                image = image[:, ::-1, :]
                self.mirror_param = (True, 'h')
            else:                         # Vertical flip
                image = image[::-1, :, :]
                self.mirror_param = (True, 'v')
        return image
        
    def do_image_jittering(self, image):
        assert image.shape[0] % self.jitter == 0
        assert image.shape[1] % self.jitter == 0
        if bool(np.random.choice(2)):
            for x in range(0, image.shape[0], self.jitter):
                for y in range(0, image.shape[1], self.jitter):
                    window = image[x:x+self.jitter, y:y+self.jitter]
                    rand = random.choice(random.choice(window))
                    window = np.array([[rand]*self.jitter]*self.jitter)
                    image[x:x+self.jitter, y:y+self.jitter] = window
        return image
        
    def apply_noise(self, image):
        if bool(np.random.choice(2)):
            noise = np.random.randint(0,50,(image.shape[0], image.shape[1]))
            zitter = np.zeros_like(image)
            zitter[:,:,1] = noise
            image = cv2.add(image, zitter)
        return image
        
    def do_random_crop(self, image, idxs=None):
        if idxs:
            return image[idxs[0]:idxs[0]+self.crop[0], idxs[1]:idxs[1]+self.crop[1]]
        
        assert self.crop[0] <= image.shape[0]
        assert self.crop[1] <= image.shape[1]
        if True:#bool(np.random.choice(2)):
            randX = np.random.choice(image.shape[0]-self.crop[0])
            randY = np.random.choice(image.shape[1]-self.crop[1])
            self.crop_param = (True, (randX, randY))
            image = image[randX:randX+self.crop[0], randY:randY+self.crop[1]]
        return image


# Following aren't ready...might be nice to polish them up and contribute them
# to fcn.berkeleyvision.org.


class IndexFileDataLayer(SemSegDataLayer):

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - dataset_dir: path to dataset's top-level directory
        - image_list: path from dataset_dir to the image-index text file
        - mean: tuple of mean values to subtract
        - data_dir: path from dataset_dir to folder containing images
        - label_dir: path from dataset_dir to folder containing labels
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for PASCAL VOC semantic segmentation.

        example

        params = dict(dataset_dir="/path/to/PASCAL/VOC2011",
            mean=(104.00698793, 116.66876762, 122.67891434),
            image_list="val")
        """
        # config
        params = eval(self.param_str)
        super(IndexFileDataLayer, self).setup(bottom, top)
        self.data_dir = params['data_dir']
        self.label_dir = params['label_dir']

    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        return Image.open('{}/{}/{}.jpg'.format(
            self.dataset_dir, self.data_dir, idx))


# replaces VOCSegDataLayer
class ImageIdxDataLayer(IndexFileDataLayer):
    """
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label samples.
        The leading singleton dimension is required by the loss.
        """
        imageFn = '{}/{}/{}.png'.format(self.dataset_dir, self.label_dir, idx)
        im = Image.open(imageFn)
        label = np.array(im, dtype=np.uint8)
        label = label[np.newaxis, ...]
        return label


# replaces SBDDSegDataLayer
class MatIdxDataLayer(IndexFileDataLayer):
    
    """
    """
    
    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label samples.
        The leading singleton dimension is required by the loss.
        """
        import scipy.io
        mat = scipy.io.loadmat('{}/{}/{}.mat'.format(self.sbdd_dir,
                                                     self.label_dir, idx))
        label = mat['GTcls'][0]['Segmentation'][0].astype(np.uint8)
        label = label[np.newaxis, ...]
        return label

# TODO add voclayer and sbddlayer
