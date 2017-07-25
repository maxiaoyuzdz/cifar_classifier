
import numpy as np
import skimage
from PIL import Image
#from pylab import cm
import random

from matplotlib import cm, colors
import scipy.misc



import matplotlib.pyplot as plt

from PIL import ImageFilter


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

def main():
    img = Image.open('/media/maxiaoyu/data/training_data/images/macaw2.jpg')
    img.load()
    #img.show()

    data = np.asarray(img)
    ds = data.shape

    print(data[1, 1, 1])

    #zoom in
    zoom_scale = np.random.random((1,))[0]
    while zoom_scale > 0.3:
        zoom_scale = np.random.random((1,))[0]


    outimg_w = int(img.width * (1 + zoom_scale))
    outimg_h = int(img.height * (1 + zoom_scale))

    outimg = img.resize((outimg_w, outimg_h))

    outimg.show()

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