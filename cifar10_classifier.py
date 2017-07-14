import sys
import time
import argparse
import torch
import torchvision

from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torch.backends.cudnn as cudnn

from datautils import *
from torch.utils.data.sampler import RandomSampler, SubsetRandomSampler
from logoperator import SaveLog

from cifarclassifier import Cifar10Classifier, Cifar10ClassifierV1

import cifarclassifier
from averagemeter import AverageMeter


parser = argparse.ArgumentParser(description='Process training arguments')
parser.add_argument('-e', '--epoch', default=300, type=int)
parser.add_argument('-mb', '--mini_batch_size', default=128, type=int)
parser.add_argument('-tb', '--test_batch_size', default=128, type=int)
parser.add_argument('-lr', '--learning_rate', default=0.05, type=float)
parser.add_argument('-mu', '--momentum', default=0.9, type=float)

parser.add_argument('-al', '--adjust_lr', default=0, type=int)
parser.add_argument('-ap', '--adjust_period', default=30, type=int)
parser.add_argument('-ar', '--adjust_rate', default=0.5, type=float)

parser.add_argument('-w', '--loader_worker', default=4, type=int)

parser.add_argument('-logdir', '--log_dir', default='/media/maxiaoyu/data/Log/')
parser.add_argument('-log', '--log_file_name', default='running.log')

parser.add_argument('-d', '--data_path', default='/media/maxiaoyu/data/training_data')
parser.add_argument('-pf', '--print_freq', default=128, type=int)



def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum()
        res.append( correct_k.mul_(100.0 / batch_size) )

    return res





def adjustLearningRatePeriodically(op, epoch):
    lr = args.learning_rate * (args.adjust_rate ** (epoch // args.adjust_period))
    print('new lr = ', lr, ' basic lr = ', args.learning_rate, ' epoch = ', epoch)
    for param_group in op.param_groups:
        param_group['lr'] = lr

def adjustLearningRateManually(op, epoch):
    if epoch >=150:
        lr = 0.01
        if epoch >= 150 and epoch < 250:
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



    #training set
    training_transforms = getTransformsForTraining()
    training_set = torchvision.datasets.CIFAR10(root=args.data_path,
                                            train=True, download=False, transform=training_transforms)
    training_set_loader = torch.utils.data.DataLoader(training_set, batch_size=args.mini_batch_size,
                                              shuffle=True, num_workers=args.loader_worker)
    # validation set
    validation_transforms = getTransformsForValidation()
    val_set = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=False,
                                          transform=validation_transforms)
    val_set_loader = torch.utils.data.DataLoader(val_set, batch_size=args.mini_batch_size,
                                              shuffle=False, num_workers=args.loader_worker)

    #test set
    testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False,
                                           download=False, transform=validation_transforms)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                             shuffle=False, num_workers=args.loader_worker)


    # prepare model
    model = cifarclassifier.__dict__['vgg11']()
    # model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    for epoch in range(args.epoch):
        running_loss = 0.0
        val_loss = 0.0
        computed_training_loss = 0
        computed_val_loss = 0

        #adjust learning rate
        adjustLearningRateControl(optimizer, epoch)

        #training
        train(training_set_loader, net, )



        # print epoch loss
        print('training loss = %.16f, validation loss = %0.16f' % (computed_training_loss, computed_val_loss))


        # epoch test
        correct = 0
        total = 0

        for data in testloader:
            images, labels = data
            outputs = net(Variable(images.cuda()))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            for j in range(args.test_batch_size):
                if labels[j] == predicted[j][0]:
                    correct += 1
        test_accurate = 100 * correct / total
        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

        #save log
        SaveLog(epoch + 1, 5000, computed_training_loss, computed_val_loss, test_accurate,
                args.log_dir + args.log_file_name )
        print('save log end')


def train(training_set_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()


    model.train()
    end = time.time()

    for i, (input, target) in enumerate(training_set_loader):
        data_time.update(time.time() - end)

        input_var, target_var = Variable(input.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(input_var)
        loss = criterion(output, target_var)
        loss.backward()
        optimizer.step()


        #transfer Variable to float Varibale
        output_value = output.float()
        loss_value = loss.float()

        #measure accuracy
        prec1 = accuracy(output_value.data, target)[0]
        losses.update(loss_value[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        #measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                  epoch, i, len(training_set_loader), batch_time=batch_time,
                  data_time=data_time, loss=losses, top1=top1))










def main():
    global args
    args = parser.parse_args()
    #print(args.log_dir + args.log_file_name)
    #testdata()
    runTraining()




if __name__ == '__main__':
    main()





