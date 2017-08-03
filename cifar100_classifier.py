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
parser.add_argument('-tb', '--test_batch_size', default=128, type=int)
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

parser.add_argument('-sa', '--save_best_allow', type=str2bool, nargs='?',
                    const=True, default="False",
                    help="Allow to save checkpoint.")


best_prec1 = 0

validation_accuracy_data = []

limit_adjust_lr_count = 0

current_limit_learning_rate = 1.0


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
    """
    print('=========')
    print(pred)
    print('---------')
    print(target)
    print('=========')
    """
    correct = pred.eq(target.view(1, -1).expand_as(pred).long())

    res = []
    correct_k = correct.sum()
    res.append(correct_k * 100.0 / batch_size)

    return res

def judgeStopTraining(op, epoch, shouldv=20):
    global validation_accuracy_data
    global limit_adjust_lr_count
    if epoch % 5 == 0:
        va = np.array(validation_accuracy_data)[::-1]
        if va.size < shouldv:
            return False
        elif va.size >= shouldv:
            max_va = va.max()
            #print(max_va)
            va_std = []
            #print(va.size)
            for index in np.arange(0, shouldv, 5):
                #print(va[index: index + 5], va[index: index + 5].std())
                va_std.append(va[index: index + 5].std())

            # key 1
            va_std_avg = np.mean(va_std)
            #print(va_std_avg)
            va_max_small = va[0:shouldv].max()
            #print(va_max_small)
            # key 2
            max_dis = np.abs(max_va - va_max_small)
            #print(max_dis)
            if va_std_avg <= 0.07 and max_dis <= 0.2:
                #print('end')
                if limit_adjust_lr_count > 1:
                    return True
                else:
                    limit_adjust_lr_count += 1
                    adjustLearningRateForceDownOneLevel(op, epoch)
                    return False
            else:
                #print('not end')
                return False

    return False


def adjustLearningRateForceDownOneLevel(op, epoch):
    global current_limit_learning_rate
    new_factor = (epoch // args.adjust_period) + 1
    lr = args.learning_rate * (args.adjust_rate ** new_factor)
    current_limit_learning_rate = lr
    print('force to use new lr = ', lr, ' basic lr = ', args.learning_rate, ' epoch = ', epoch)
    for param_group in op.param_groups:
        param_group['lr'] = lr


def adjustLearningRatePeriodically(op, epoch):
    global current_limit_learning_rate
    lr = args.learning_rate * (args.adjust_rate ** (epoch // args.adjust_period))
    if lr < current_limit_learning_rate:
        print('auto adjust lr = ', lr, ' basic lr = ', args.learning_rate, ' epoch = ', epoch)
        for param_group in op.param_groups:
            param_group['lr'] = lr
    else:
        print('current limit lr is lower')


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
        adjustLearningRateManually(op, epoch)
        adjustLearningRateManually(op, epoch)
    elif args.adjust_lr == 1:
        adjustLearningRatePeriodically(op, epoch)
    elif args.adjust_lr == 2:
        adjustLearningRateManually(op, epoch)




def runTraining():
    global best_prec1
    global validation_accuracy_data

    training_start_time = time.time()
    # prepare model, select from args
    net = cifar100cnn.__dict__[args.arch]()
    # data parallel 1
    #net.features = torch.nn.DataParallel(net.features)
    net.cuda()

    # data parallel 2
    #net = torch.nn.DataParallel(net).cuda()
    #cudnn.benchmark = True

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
    training_set = torchvision.datasets.CIFAR100(root=args.data_path,
                                                train=True, download=False, transform=training_transforms)
    training_set_loader = torch.utils.data.DataLoader(training_set, batch_size=args.mini_batch_size,
                                                      shuffle=True, num_workers=args.loader_worker)

    # validation set
    validation_transforms = getTransformsForValidation()
    val_set = torchvision.datasets.CIFAR100(root=args.data_path,
                                           train=False, download=False, transform=validation_transforms)
    val_set_loader = torch.utils.data.DataLoader(val_set, batch_size=args.mini_batch_size,
                                                 shuffle=False, num_workers=args.loader_worker)

    for epoch in range(args.start_epoch, args.epoch):
        # judge to stop training
        if judgeStopTraining(optimizer, epoch):
            print('end training')
            break



        # epoch time
        epoch_start_time = time.time()
        # adjust learning rate
        adjustLearningRateControl(optimizer, epoch)

        # training
        training_loss, training_accuracy = train(training_set_loader, net, criterion, optimizer, epoch)

        # evaluate on validation set
        val_loss, val_accuracy = validate(val_set_loader, net, criterion)
        # save val_accuracy
        validation_accuracy_data.append(val_accuracy)
        # check best
        is_best = val_accuracy > best_prec1
        best_prec1 = max(val_accuracy, best_prec1)

        # save log
        SaveLog(epoch + 1, 5000, training_loss, val_loss, training_accuracy, val_accuracy,
                args.log_dir + args.log_file_name)
        #save running model
        if args.save_best_allow:
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
        have_run_time = (epoch_end_time - training_start_time) / 60
        left_time = epoch_running_time * (args.epoch - epoch)
        print('epoch : {0}, running time : {1:.2f}m , have used : {2:.2f}m, left estimate : {3:.2f}m'.
              format(epoch, epoch_running_time, have_run_time, left_time))
        print('train loss = {0:.2f}, val loss = {1:.2f}, train ac = {2:.2f}, val ac = {3:.2f}'.
              format(training_loss, val_loss, training_accuracy, val_accuracy))


    training_end_time = time.time()
    training_running_time = training_end_time - training_start_time
    print('total running time = ', training_running_time / 60, ' mins')


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
    if args.args_check_allow is True:
        input('Press Enter to Continue')
    runTraining()


if __name__ == '__main__':
    main()
