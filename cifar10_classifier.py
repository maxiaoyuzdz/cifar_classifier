import torch
import torchvision
#import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datautils import getTransform

from torch.utils.data.sampler import RandomSampler, SubsetRandomSampler

def imshow(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    np.transpose(npimg, (1, 2, 0))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def plotshow(x, y):
    x = np.arange(0, 3 * np.pi, 0.1)
    y = np.sin(x)

    plt.plot(x, y)
    plt.show()

def count(iter):
    return sum(1 for _ in iter)


class MxyCifarClassifierNet(nn.Module):
    def __init__(self):
        super(MxyCifarClassifierNet, self).__init__()
        #layer setting
        self.conv1 = nn.Conv2d(3, 9, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(9, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)



    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

MINI_BATCH_SIZE = 10
EPOCH_NUM = 2

transform = getTransform()

trainset = torchvision.datasets.CIFAR10(root='/media/maxiaoyu/datastore/training_data',
                                        train=True, download=True, transform=transform)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=MINI_BATCH_SIZE,
                                          shuffle=True, num_workers=2)



TEST_BATCH_SIZE = 4
testset = torchvision.datasets.CIFAR10(root='/media/maxiaoyu/datastore/training_data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=TEST_BATCH_SIZE,
                                         shuffle=False, num_workers=2)


"""
# print(count(trainset))
# tempdata = enumerate(trainloader)
# print(count(tempdata)) 5000


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# dataiter = iter(trainloader)
# images, labels = dataiter.next()


# show img, but is a block operation
# imshow(torchvision.utils.make_grid(images))


# plotshow(10, 20)


"""
net = MxyCifarClassifierNet()
net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(EPOCH_NUM):
    running_loss = 0.0
    val_loss = 0.0
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
                print('[%d, %5d] training loss: %.3f' %
                      (epoch + 1, index + 1, running_loss / 2000))
                running_loss = 0.0

        if index >= 4000:
            inputs, targets = data
            inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
            output = net(inputs)
            loss = criterion(output, targets)
            val_loss += loss.data[0]
            if index % 1000 == 999:
                print('[%d, %5d] validation loss: %.3f' %
                      (epoch + 1, index + 1, val_loss / 1000))
                val_loss = 0.0


print('Finished Training')

print('start test')

"""
dataiter = iter(testloader)
images, labels = dataiter.next()
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(TEST_BATCH_SIZE)))
test_outputs = net(Variable(images.cuda()))
_, predicted = torch.max(test_outputs.data, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j][0]]
                              for j in range(TEST_BATCH_SIZE)))
"""
correct = 0
total = 0

for data in testloader:
    images, labels = data
    outputs = net(Variable(images.cuda()))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    local_count = 0

    for j in range(TEST_BATCH_SIZE):
        if(labels[j] == predicted[j][0]):
            correct += 1

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


