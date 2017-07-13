import torch.nn as nn
import torch.nn.functional as F
import math

# official tutorial model
class Cifar10Classifier(nn.Module):
    def __init__(self):
        super(Cifar10Classifier, self).__init__()
        # layer setting
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

# update version 1
class Cifar10ClassifierV1(nn.Module):
    def __init__(self):
        super(Cifar10ClassifierV1, self).__init__()
        # layer setting
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# dymatic creating model

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
}


def make_layers(cfg):
    print(cfg)
    layers = []
    in_channels = 3
    for v in cfg:
        if v is 'M':
            print('a max pool')
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            print('conv2d in = ', in_channels, '  out = ', v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)

class CNNModel(nn.Module):
    def __init__(self, features):
        super(CNNModel, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        #print(' parameter = ', x.size())
        x = x.view(x.size(0), -1)
        #print(' parameter = ', x.size())
        x = self.classifier(x)
        return x


def vgg11():
    net = CNNModel(make_layers(cfg['A']))
    return net

def vggtest():
    return 'asdf'




