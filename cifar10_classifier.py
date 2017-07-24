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


import cifarclassifier
from averagemeter import AverageMeter

parser = argparse.ArgumentParser(description='Process training arguments')

parser.add_argument('-a', '--arch', default='vgg11')

parser.add_argument('-r', '--resume', type=str2bool, nargs='?',
                    const=True, default="False",
                    help="Activate nice mode.")
parser.add_argument('-rp', '--resume_path', default='/media/maxiaoyu/checkpoint/')
parser.add_argument('-rf', '--resume_file', default='_checkpoint.pth.tar')

parser.add_argument('-st', '--start_epoch', default=0, type=int)
parser.add_argument('-e', '--epoch', default=300, type=int)
parser.add_argument('-mb', '--mini_batch_size', default=128, type=int)
parser.add_argument('-tb', '--test_batch_size', default=128, type=int)
parser.add_argument('-lr', '--learning_rate', default=0.05, type=float)
parser.add_argument('-mu', '--momentum', default=0.9, type=float)

parser.add_argument('-wda', '--weight_decay_allow', type=str2bool, nargs='?',
                    const=True, default="False",
                    help="Activate L2 Regularization.")
parser.add_argument('-wd', '--weight_decay', default=5e-4, type=float)

parser.add_argument('-al', '--adjust_lr', default=1, type=int)
parser.add_argument('-ap', '--adjust_period', default=30, type=int)
parser.add_argument('-ar', '--adjust_rate', default=0.5, type=float)

parser.add_argument('-logdir', '--log_dir', default='/media/maxiaoyu/data/Log/')
parser.add_argument('-log', '--log_file_name', default='running.log')
parser.add_argument('-d', '--data_path', default='/media/maxiaoyu/data/training_data')
parser.add_argument('-w', '--loader_worker', default=4, type=int)

parser.add_argument('-pa', '--print_allow', type=str2bool, nargs='?',
                    const=True, default="False",
                    help="Activate Print detail in running time.")
parser.add_argument('-pf', '--print_freq', default=128, type=int)

parser.add_argument('-sp', '--save_model_path', default='/media/maxiaoyu/data/checkpoint/')
parser.add_argument('-sf', '--save_model_file', default='_checkpoint.pth.tar')
parser.add_argument('-sbf', '--save_best_model_file', default='_best_checkpoint.pth.tar')


best_prec1 = 0
training_loss_array = []
validation_loss_array = []


def saveCheckPoint(state, is_best):
    torch.save(state, args.save_model_path + args.arch + args.save_model_file)
    if is_best:
        shutil.copy(args.save_model_path + args.arch + args.save_model_file,
                    args.save_model_path + args.arch + args.save_best_model_file)


def accuracy(output, target, topk=(1,)):
    """use cuda tensor, parameters output and target are cuda tensor"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred).long())

    res = []
    correct_k = correct.sum()
    res.append(correct_k * 100.0 / batch_size)

    return res


def adjustLearningRatePeriodically(op, epoch):
    lr = args.learning_rate * (args.adjust_rate ** (epoch // args.adjust_period))
    print('new lr = ', lr, ' basic lr = ', args.learning_rate, ' epoch = ', epoch)
    for param_group in op.param_groups:
        param_group['lr'] = lr


def adjustLearningRateManually(op, epoch):
    if epoch >= 150:
        lr = 0.01
        if 150 <= epoch < 250:
            lr = 0.01
        elif epoch >= 250:
            lr = 0.001
        for param_group in op.param_groups:
            param_group['lr'] = lr


def adjustLearningRateControl(op, epoch):
    if args.adjust_lr == 0:
        pass
    elif args.adjust_lr == 1:
        adjustLearningRatePeriodically(op, epoch)
    elif args.adjust_lr == 2:
        adjustLearningRateManually(op, epoch)


def runTraining():
    global best_prec1
    start_time = time.time()
    # prepare model, select from args
    net = cifarclassifier.__dict__[args.arch]()
    # model.features = torch.nn.DataParallel(model.features)
    net.cuda()
    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().cuda()
    if args.weight_decay_allow:
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum,
                              weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum)

    # resume
    if args.resume:
        if os.path.isfile(args.resume_path + args.resume_file):
            print('=> loading checkpoint ')
            checkpoint = torch.load(args.resume_path + args.resume_file)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}".format(checkpoint['epoch']))
        else:
            print('=> no checkpoint found ')

    # training set
    training_transforms = getTransformsForTraining()
    training_set = torchvision.datasets.CIFAR10(root=args.data_path,
                                                train=True, download=False, transform=training_transforms)
    training_set_loader = torch.utils.data.DataLoader(training_set, batch_size=args.mini_batch_size,
                                                      shuffle=True, num_workers=args.loader_worker)

    # validation set
    validation_transforms = getTransformsForValidation()
    val_set = torchvision.datasets.CIFAR10(root=args.data_path,
                                           train=False, download=False, transform=validation_transforms)
    val_set_loader = torch.utils.data.DataLoader(val_set, batch_size=args.mini_batch_size,
                                                 shuffle=False, num_workers=args.loader_worker)

    for epoch in range(args.start_epoch, args.epoch):
        # epoch time
        epoch_start_time = time.time()
        # adjust learning rate
        adjustLearningRateControl(optimizer, epoch)

        # training
        training_loss, training_accuracy = train(training_set_loader, net, criterion, optimizer, epoch)

        # evaluate on validation set
        val_loss, val_accuracy = validate(val_set_loader, net, criterion)

        is_best = val_accuracy > best_prec1
        best_prec1 = max(val_accuracy, best_prec1)

        # save log
        SaveLog(epoch + 1, 5000, training_loss, val_loss, training_accuracy, val_accuracy,
                args.log_dir + args.log_file_name)
        #save running model
        saveCheckPoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': net.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)
        print('save log end')
        epoch_end_time = time.time()
        epoch_running_time = (epoch_end_time - epoch_start_time) / 60
        left_time = epoch_running_time * (args.epoch - epoch)
        print('epoch :', epoch, ' , running time :', epoch_running_time, 'm, eft estimate :', left_time, 'm')

    end_time = time.time()
    running_time = end_time - start_time
    print('running time = ', running_time / 60, ' mins')


def train(training_set_loader, model, criterion, optimizer, epoch):
    """train a epoch, return average loss and accuracy"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    model.train()
    end = time.time()

    for i, (input, target) in enumerate(training_set_loader):
        data_time.update(time.time() - end)

        input_cuda_var, target_cuda_var = Variable(input.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output_cuda_var = model(input_cuda_var)
        loss = criterion(output_cuda_var, target_cuda_var)
        loss.backward()
        optimizer.step()

        # transfer Variable to float Varibale
        output_cuda_var_float = output_cuda_var.float()
        loss = loss.float()

        # measure accuracy use Variable's Tensor
        prec1 = accuracy(output_cuda_var_float.data, target_cuda_var.data)[0]
        losses.update(loss.data[0], input.size(0))
        accuracies.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.print_allow and i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(training_set_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=accuracies))

    # return average loss and accuracy
    return losses.avg, accuracies.avg


def validate(val_set_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    model.eval()
    end = time.time()

    for i, (input, target) in enumerate(val_set_loader):
        input_cuda_var, target_cuda_var = Variable(input.cuda()), Variable(target.cuda())

        output_cuda_var = model(input_cuda_var)
        loss = criterion(output_cuda_var, target_cuda_var)

        # transfer Variable to float Variable
        output_cuda_var_float = output_cuda_var.float()
        loss = loss.float()

        # measure accuracy
        prec1 = accuracy(output_cuda_var_float.data, target_cuda_var.data)[0]
        losses.update(loss.data[0], input.size(0))
        accuracies.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.print_allow and i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_set_loader), batch_time=batch_time,
                loss=losses, top1=accuracies))

    print('val * Prec@1 {top1.avg: .3f}'
          .format(top1=accuracies))

    # return average loss and accuracy
    return losses.avg, accuracies.avg


def main():
    global args
    args = parser.parse_args()
    #check parameters
    print('=====  Please Check Parameters  =====')
    print(args)
    print('=====================================')
    input('Press Enter to Continue')
    runTraining()


if __name__ == '__main__':
    main()
