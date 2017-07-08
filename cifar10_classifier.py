import sys
import argparse
import torch
import torchvision

from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from datautils import getTransform
from torch.utils.data.sampler import RandomSampler, SubsetRandomSampler
from logoperator import SaveLog

from cifarclassifier import Cifar10Classifier, Cifar10ClassifierV1


parser = argparse.ArgumentParser(description='Process training arguments')
parser.add_argument('-e', '--epoch', default=300, type=int)
parser.add_argument('-mb', '--mini_batch_size', default=10, type=int)
parser.add_argument('-tb', '--test_batch_size', default=4, type=int)
parser.add_argument('-lr', '--learning_rate', default=0.01, type=float)
parser.add_argument('-logdir', '--log_dir', default='/media/maxiaoyu/datastore/Log/')
parser.add_argument('-log', '--log_file_name', default='running.log')


def adjustlearningrate(op, blr, epoch):
    lr = blr * 0.1 ** (epoch // 30)
    print('new lr = ', lr, ' old lr = ', blr, ' epoch = ',epoch)
    for param_group in op.param_groups:
        param_group['lr'] = lr


def runtraining(epoch_num, mini_batch_size, test_batch_size, learning_rate, log_path):
    transform = getTransform()
    trainset = torchvision.datasets.CIFAR10(root='/media/maxiaoyu/datastore/training_data',
                                            train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=mini_batch_size,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='/media/maxiaoyu/datastore/training_data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                             shuffle=False, num_workers=2)

    net = Cifar10ClassifierV1()
    net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(epoch_num):
        running_loss = 0.0
        val_loss = 0.0
        computed_training_loss = 0
        computed_val_loss = 0
        #adjust learning rate
        adjustlearningrate(optimizer, learning_rate, epoch)

        #training
        for index, data in enumerate(trainloader):
            if index < 4000:
                inputs, targets = data
                inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())

                optimizer.zero_grad()
                output = net(inputs)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.data[0]

                if index % 2000 == 1999:
                    print('[%d, %5d] training loss: %.16f' %
                          (epoch + 1, index + 1, running_loss / 2000))
                    computed_training_loss = running_loss / 2000
                    running_loss = 0.0

            if index >= 4000:
                inputs, targets = data
                inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
                output = net(inputs)
                loss = criterion(output, targets)
                val_loss += loss.data[0]
                if index % 1000 == 999:
                    print('[%d, %5d] validation loss: %.16f' %
                          (epoch + 1, index + 1, val_loss / 1000))
                    computed_val_loss = val_loss / 1000
                    val_loss = 0.0

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

            for j in range(test_batch_size):
                if labels[j] == predicted[j][0]:
                    correct += 1
        test_accurate = 100 * correct / total
        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

        #save log
        SaveLog(epoch + 1, 5000, computed_training_loss, computed_val_loss, test_accurate, log_path)
        print('save log end')





def main():
    global args
    args = parser.parse_args()
    #print(args.log_dir + args.log_file_name)

    runtraining(args.epoch, args.mini_batch_size, args.test_batch_size, args.learning_rate, args.log_dir + args.log_file_name)


if __name__ == '__main__':
    main()





