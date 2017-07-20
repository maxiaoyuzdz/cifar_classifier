
import numpy as np
import torch
import torchvision
import time



from datautils import *

import matplotlib.pyplot as plt
from PIL import Image






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

def rotate90Transform(img):
    if np.random.randint(0, 2) == 1:
        img = img.rotate(90)
    return img


def rotate180Transform(img):
    if np.random.randint(0, 2) == 1:
        img = img.rotate(180)
    return img


def noiseTransform(img):
    if np.random.randint(0, 2) == 1:
        noise = np.random.randint(0, 50, (img.height, img.width))  # design jitter/noise here
        zitter = np.zeros_like(img)
        zitter[:, :, 1] = noise
        noise_added = img + zitter
        return noise_added
    return img


def getTransformsForTest():
    return transforms.Compose([
        transforms.Lambda(lambda x: verticalFlipTransform(x)),
        transforms.Lambda(lambda x: rotate90Transform(x)),
        transforms.Lambda(lambda x: rotate180Transform(x)),
        transforms.Lambda(lambda x: noiseTransform(x)),
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
                                                  shuffle=False, num_workers=4)

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