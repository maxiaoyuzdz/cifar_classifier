
import numpy as np
import skimage
from PIL import Image
import random


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

def main():
    img = Image.open('/media/maxiaoyu/data/training_data/images/macaw2.jpg')
    img.load()
    img.show()

    data = np.asarray(img)
    #print('shpe = ', data.shape)
    #noise = data + 100. * np.random.randn(data.shape)
    #noise = np.random.normal(0, 0.1, (data.shape[0], data.shape[1], data.shape[2])) + data

    #data2 = skimage.util.random_noise(data, mode='gaussian', seed=0, clip=True)
    #print('shape 2 = ', data2.shape)
    #outimg = Image.fromarray(noise, "RGBA")

    # work 1
    #outimg = sp_noise(data, 0.05)

    outimg = noisy('speckle', data)

    outimg = Image.fromarray(outimg, "RGB")

    outimg.show()


if __name__ == '__main__':
    main()