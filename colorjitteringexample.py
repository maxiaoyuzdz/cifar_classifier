
import numpy as np
import torch
import torchvision



from datautils import *

import matplotlib.pyplot as plt






def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    print(inp.shape)

    plt.imshow(inp)

    if title is not None:
        plt.title(title)

    plt.pause(0.001)





def getTransformsForTest():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def main():
    print('ok')

    plt.ion()

    #load data
    transforms = getTransformsForTest()
    data_set = torchvision.datasets.CIFAR10(root='/media/maxiaoyu/data/training_data',
                                                train=False, download=False, transform=transforms)
    data_set_loader = torch.utils.data.DataLoader(data_set, batch_size=1,
                                                      shuffle=False, num_workers=4)


    #inputs, targets = next(iter(data_set_loader))

    for i, (input, target) in enumerate(data_set_loader):
        out = torchvision.utils.make_grid(input)
        imshow(out, title=[x for x in target])
        break

    print('end')






if __name__ == '__main__':
    main()