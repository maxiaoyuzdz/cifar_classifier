
import numpy as np
import skimage
from PIL import Image
#from pylab import cm
import random

from matplotlib import cm, colors
import scipy.misc



import matplotlib.pyplot as plt

from PIL import ImageFilter
import math
import colorsys


def noisy(noise_typ,image):

    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        #var = 0.1
        #sigma = var**0.5
        gauss = np.random.normal(mean,1,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        print(noisy.shape)
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = image
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = image + image * gauss
        return noisy

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


def zoomTransform(img):
    zoom_type = np.random.randint(0, 3)
    if zoom_type == 0:
        return img
    # zoom in
    elif zoom_type == 1:
        zoom_scale = np.random.random((1, ))[0]
        outimg = img.resize((45, 45))
        return outimg
    return img


def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b

def rgb2hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    return h, s, v


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


def shift_hue(arr,hout):
    hsv=rgb_to_hsv(arr)
    hsv[...,0]=hout
    rgb=hsv_to_rgb(hsv)
    return rgb

def main():
    img = Image.open('/media/maxiaoyu/data/training_data/images/macaw2.jpg')
    img.load()
    #img.show()

    data = np.asarray(img)
    ds = data.shape

    print(data[1, 1, 1])

    #zoom in
    """
    zoom_scale = np.random.random((1,))[0]
    while zoom_scale > 0.3:
        zoom_scale = np.random.random((1,))[0]


    outimg_w = int(img.width * (1 + zoom_scale))
    outimg_h = int(img.height * (1 + zoom_scale))

    outimg = img.resize((outimg_w, outimg_h))

    outimg.show()
    """


    # hsv operation
    random_factor2 = random.uniform(0.7, 1.4)
    random_factor3 = random.uniform(-0.1, 1.0)
    random_factor4 = random.uniform(-0.1, 1.0)
    hsv = rgb_to_hsv(data)
    print(hsv.shape)
    """
    for w in range(0, hsv.shape[0]):
        for h in range(0, hsv.shape[1]):
            s = hsv[w, h, 1]
            s = s * random_factor2 + random_factor3
            hsv[w, h, 1] = s

            v = hsv[w, h, 2]
            v = v * random_factor2 + random_factor3
            hsv[w, h, 2] = v
    """
    #s
    hsv[:, :, 1] *= random_factor2
    hsv[:, :, 1] += random_factor3
    #v
    hsv[:, :, 2] *= random_factor2
    hsv[:, :, 2] += random_factor3
    #h
    hsv[:, :, 0] += random_factor4

    rgbimg = hsv_to_rgb(hsv)
    outimg = Image.fromarray(rgbimg, mode="RGB")
    outimg.show('one')

    data2 = np.zeros(ds)
    for wi in range(0, ds[0]):
        for hi in range(0, ds[1]):
            r = data[wi, hi, 0]
            g = data[wi, hi, 1]
            b = data[wi, hi, 2]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            h += random_factor4

            s *= random_factor2
            s += random_factor3

            v *= random_factor2
            v += random_factor3
            rn, gn, bn = colorsys.hsv_to_rgb(h, s, v)
            #rn *= 255
            #gn *= 255
            #bn *= 255
            data2[wi, hi, 0] = rn
            data2[wi, hi, 1] = gn
            data2[wi, hi, 2] = bn

    outimg2 = Image.fromarray(data2, mode="RGB")
    outimg2.show()

    #ci = scipy.misc.toimage(data2, cmin=0.0, cmax=1.0)
    #ci.show()


    print('show end')

    print('epoch : {0}, , running time : {1:.2f}m , left estimate : {2:.2f}m'.format(1, 0.345,
                                                                                     0.456))

    #blur
    #outimg = img.filter(ImageFilter.BLUR)
    #outimg.show()


    #noise
    #data2 = skimage.util.random_noise(data, mode='salt', amount=0.15)
    #ds2 = data2.shape

    #ci = scipy.misc.toimage(data2, cmin=0.0, cmax=1.0)
    #ci.show()


    #outimg = Image.fromarray(np.uint8(cm.gist_earth(data2)*255))
    #outimg = Image.fromarray(, mode="RGB")#.convert('RGB')
    #outimg.show()

    #plt.imshow(data2)
    #plt.show()


if __name__ == '__main__':
    main()