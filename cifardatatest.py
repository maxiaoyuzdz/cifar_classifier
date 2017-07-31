
import sys
import os
import time
import argparse
import shutil
import numpy as np
import torch
import torchvision

from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torch.backends.cudnn as cudnn

from datautils import *
from torch.utils.data.sampler import RandomSampler, SubsetRandomSampler
from logoperator import SaveLog


import cifar100cnn
from averagemeter import AverageMeter

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



parser = argparse.ArgumentParser(description='Process training arguments')

parser.add_argument('-a', '--arch', default='vgg16_bn')

parser.add_argument('-r', '--resume', type=str2bool, nargs='?',
                    const=True, default="False",
                    help="Activate nice mode.")
parser.add_argument('-rp', '--resume_path', default='/media/maxiaoyu/checkpoint/cifar100/')
parser.add_argument('-rf', '--resume_file', default='_checkpoint.pth.tar')

parser.add_argument('-st', '--start_epoch', default=0, type=int)
parser.add_argument('-e', '--epoch', default=300, type=int)
parser.add_argument('-mb', '--mini_batch_size', default=128, type=int)
parser.add_argument('-tb', '--test_batch_size', default=4, type=int)
parser.add_argument('-lr', '--learning_rate', default=0.1, type=float)
parser.add_argument('-mu', '--momentum', default=0.9, type=float)

parser.add_argument('-wda', '--weight_decay_allow', type=str2bool, nargs='?',
                    const=True, default="True",
                    help="Activate L2 Regularization.")

parser.add_argument('-wd', '--weight_decay', default=5e-4, type=float)

parser.add_argument('-al', '--adjust_lr', default=1, type=int)
parser.add_argument('-ap', '--adjust_period', default=30, type=int)
parser.add_argument('-ar', '--adjust_rate', default=0.5, type=float)

parser.add_argument('-logdir', '--log_dir', default='/media/maxiaoyu/data/Log/cifar100/')
parser.add_argument('-log', '--log_file_name', default='running.log')
parser.add_argument('-d', '--data_path', default='/media/maxiaoyu/data/training_data')
parser.add_argument('-w', '--loader_worker', default=4, type=int)

parser.add_argument('-pa', '--print_allow', type=str2bool, nargs='?',
                    const=True, default="False",
                    help="Activate Print detail in running time.")
parser.add_argument('-pf', '--print_freq', default=128, type=int)

parser.add_argument('-sp', '--save_model_path', default='/media/maxiaoyu/data/checkpoint/cifar100/')
parser.add_argument('-sf', '--save_model_file', default='_checkpoint.pth.tar')
parser.add_argument('-sbf', '--save_best_model_file', default='_best_checkpoint.pth.tar')
# args parameters check
parser.add_argument('-ac', '--args_check_allow', type=str2bool, nargs='?',
                    const=True, default="False",
                    help="Activate Print detail in running time.")


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

def main():
    global args
    args = parser.parse_args()
    # check parameters
    print('=====  Please Check Parameters  =====')
    print(args)
    print('=====================================')
    if args.args_check_allow is True:
        input('Press Enter to Continue')

    # training set
    training_transforms = getTransformsForTraining()
    training_set = torchvision.datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=training_transforms)
    training_set_loader = torch.utils.data.DataLoader(training_set, batch_size=args.mini_batch_size,
                                                      shuffle=True, num_workers=args.loader_worker)

    # validation set
    validation_transforms = getTransformsForValidation()
    val_set = torchvision.datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=validation_transforms)
    val_set_loader = torch.utils.data.DataLoader(val_set, batch_size=args.test_batch_size,
                                                 shuffle=False, num_workers=args.loader_worker)

    for i, (indata, target) in enumerate(val_set_loader):
        out = torchvision.utils.make_grid(indata)
        # out = imgJitter(out)
        # print('img shape = ', out.shape)
        #imshow(out)
        # input('enter to continue')
        print(target)

        break


if __name__ == '__main__':
    main()