import torchvision.transforms as transforms
import argparse
import numpy as np
from PIL import Image
import random
import skimage
import scipy.misc
from PIL import ImageFilter



def rgb_to_hsv(rgb):
    # Translated from source of colorsys.rgb_to_hsv
    # r,g,b should be a numpy arrays with values between 0 and 255
    # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
    rgb = rgb.astype('float')
    hsv = np.zeros_like(rgb)
    # in case an RGBA array was passed, just copy the A channel
    hsv[..., 3:] = rgb[..., 3:]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb[..., :3], axis=-1)
    minc = np.min(rgb[..., :3], axis=-1)
    # v
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select(
        [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    return hsv
def hsv_to_rgb(hsv):
    # Translated from source of colorsys.hsv_to_rgb
    # h,s should be a numpy arrays with values between 0.0 and 1.0
    # v should be a numpy array with values between 0.0 and 255.0
    # hsv_to_rgb returns an array of uints between 0 and 255.
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')




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


def hsvTransform(img):
    if np.random.randint(0, 2) == 1:
        random_factor2 = random.uniform(0.7, 1.4)
        random_factor3 = random.uniform(-0.1, 1.0)
        random_factor4 = random.uniform(-0.1, 1.0)
        data = np.asarray(img)
        hsv = rgb_to_hsv(data)
        # s
        hsv[:, :, 1] *= random_factor2
        hsv[:, :, 1] += random_factor3
        # v
        hsv[:, :, 2] *= random_factor2
        hsv[:, :, 2] += random_factor3
        # h
        hsv[:, :, 0] += random_factor4

        rgb = hsv_to_rgb(hsv)
        outimg = Image.fromarray(rgb, mode="RGB")
        return outimg
    return img


def getTransform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def getTransformsForVGGTraining():
    return transforms.Compose([
        #transforms.Lambda(lambda x: hsvTransform(x)),
        #transforms.Lambda(lambda x: verticalFlipTransform(x)),
        #transforms.Lambda(lambda x: rotateTransform(x)),
        #transforms.Lambda(lambda x: noiseTransform(x)),
        transforms.Scale(224),
        transforms.RandomCrop(224, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def getTransformsForVGGValidation():
    return transforms.Compose([
        transforms.Scale(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def getTransformsForTraining():
    return transforms.Compose([
        #transforms.Lambda(lambda x: hsvTransform(x)),
        #transforms.Lambda(lambda x: verticalFlipTransform(x)),
        #transforms.Lambda(lambda x: rotateTransform(x)),
        #transforms.Lambda(lambda x: noiseTransform(x)),
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
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