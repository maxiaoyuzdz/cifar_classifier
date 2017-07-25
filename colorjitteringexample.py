
import numpy as np
import torch
import torchvision
import time



from datautils import *

import matplotlib.pyplot as plt
from PIL import Image


import random

import skimage
import scipy.misc
from PIL import ImageFilter






def imshow(img, title=None):
    img = img.numpy()
    img = img.transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    #print(img.shape)

    plt.imshow(img)

    if title is not None:
        plt.title(title)

    plt.pause(0.001)



def imgJitter(img):
    """ img is Tensor, need to transfer numpy ndarray"""
    img = img.numpy()
    img = img.transpose((1, 2, 0))
    h, w, c = img.shape
    print(h)
    print(img.shape)
    """
    noise = np.random.randint(0, 50, (h, w))  # design jitter/noise here
    zitter = np.zeros_like(img)
    zitter[:, :, 1] = noise
    print(zitter.shape)
    noise_added = img + zitter
    print(noise_added.shape)

    combined = np.vstack((img[:17, :, :], noise_added[17:, :, :]))
"""
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    RGBshifted = np.dstack((
        np.roll(R, 2, axis=0),
        np.roll(G, 1, axis=0),
        np.roll(B, 2, axis=0)
    ))


    return RGBshifted


def myTransform(x):

    #
    if np.random.randint(0, 2) == 1:
        x = x.rotate(45)
    #x = Image.new("RGB", (x.width, x.height), "white")

    return x


def verticalFlipTransform(img):
    if np.random.randint(0, 2) == 1:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img

def rotateTransform(img):
    if np.random.randint(0, 2) == 1:
        if np.random.randint(0, 2) == 1:
            img = img.rotate(90)
        else:
            img = img.rotate(180)
    return img


def sp_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def saltpepperNoiseTransform(img):
    """
    if np.random.randint(0, 2) == 1:
        noise = np.random.randint(0, 50, (img.height, img.width))  # design jitter/noise here
        zitter = np.zeros_like(img)
        zitter[:, :, 1] = noise
        noise_added = img + zitter
        return noise_added
    """
    """
    data = np.asarray(img, dtype="int32")
    print('shpe = ', data.shape)
    data = skimage.util.random_noise(data, mode='gaussian', seed=None, clip=True)
    print('shape 2 = ', data.shape)
    outimg = Image.fromarray(data, "RGB")
    """
    data = np.asarray(img)
    outimg = sp_noise(data, 0.05)
    outimg = Image.fromarray(outimg)
    return outimg

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


def getTransformsForTest():
    return transforms.Compose([
        #transforms.Lambda(lambda x: verticalFlipTransform(x)),
        #transforms.Lambda(lambda x: rotateTransform(x)),
        #transforms.Lambda(lambda x: noiseTransform(x)),

        transforms.Lambda(lambda x: zoomTransform(x)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        # blur
        #transforms.Lambda(lambda x: blurTransform(x)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])



def imgtest1():
    # load data
    transforms = getTransformsForTest()
    data_set = torchvision.datasets.CIFAR10(root='/media/maxiaoyu/data/training_data',
                                            train=False, download=False, transform=transforms)
    data_set_loader = torch.utils.data.DataLoader(data_set, batch_size=4,
                                                  shuffle=False, num_workers=1)

    # inputs, targets = next(iter(data_set_loader))

    for i, (indata, target) in enumerate(data_set_loader):
        out = torchvision.utils.make_grid(indata)
        #out = imgJitter(out)
        #print('img shape = ', out.shape)
        imshow(out)
        #input('enter to continue')
        break

def main():
    print('ok')

    plt.ion()

    imgtest1()

    print('end')






if __name__ == '__main__':
    main()