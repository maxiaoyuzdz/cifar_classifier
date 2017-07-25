import torchvision.transforms as transforms
import argparse
import numpy as np
from PIL import Image
import random
import skimage
import scipy.misc
from PIL import ImageFilter


def verticalFlipTransform(img):
    if np.random.randint(0, 2) == 1:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img

def rotateTransform(img):
    rotate_type = np.random.randint(0, 3)
    if rotate_type == 0:
        return img
    elif rotate_type == 1:
        img = img.rotate(90)
    elif rotate_type == 2:
        img = img.rotate(180)
    return img

def noiseTransform(img):
    noise_type = np.random.randint(0, 5)

    if noise_type == 0:
        return img

    data = np.asarray(img)
    if noise_type == 1:
        #pepper s&p salt
        data2 = skimage.util.random_noise(data, mode='s&p', amount=0.15)
        ci = scipy.misc.toimage(data2, cmin=0.0, cmax=1.0)
        return ci
    elif noise_type == 2:
        data2 = skimage.util.random_noise(data, mode='gaussian')
        ci = scipy.misc.toimage(data2, cmin=0.0, cmax=1.0)
        return ci
    elif noise_type == 3:
        data2 = skimage.util.random_noise(data, mode='localvar')
        ci = scipy.misc.toimage(data2, cmin=0.0, cmax=1.0)
        return ci
    elif noise_type == 4:
        data2 = skimage.util.random_noise(data, mode='poisson')
        ci = scipy.misc.toimage(data2, cmin=0.0, cmax=1.0)
        return ci


def blurTransform(img):
    if np.random.randint(0, 2) == 1:
        outimg = img.filter(ImageFilter.BLUR)
        return outimg
    return img


def zoomTransform(img):
    zoom_type = np.random.randint(0, 3)
    if zoom_type == 0:
        return img
    # zoom in
    elif zoom_type == 1:
        zoom_scale = np.random.random((1,))[0]
        while zoom_scale > 0.3:
            zoom_scale = np.random.random((1,))[0]

        outimg_w = int(img.width * (1 + zoom_scale))
        outimg_h = int(img.height * (1 + zoom_scale))
        outimg = img.resize((outimg_w, outimg_h))
        return outimg
    # zoom out
    elif zoom_type == 2:
        zoom_scale = np.random.random((1,))[0]
        while zoom_scale > 0.3:
            zoom_scale = np.random.random((1,))[0]

        outimg_w = int(img.width * (1 - zoom_scale))
        outimg_h = int(img.height * (1 - zoom_scale))
        outimg = img.resize((outimg_w, outimg_h))
        return outimg
    return img


def getTransform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def getTransformsForTraining():
    return transforms.Compose([
        transforms.Lambda(lambda x: verticalFlipTransform(x)),
        transforms.Lambda(lambda x: rotateTransform(x)),
        #transforms.Lambda(lambda x: noiseTransform(x)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def getTransformsForValidation():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])



def count(iter):
    return sum(1 for _ in iter)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2array(s):
    return s.split(',')