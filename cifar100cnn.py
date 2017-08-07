import torch.nn as nn
import torch.nn.functional as F
import math

# dymatic creating model

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_layers(cfg, batch_norm=False):
    print(cfg)
    layers = []
    in_channels = 3
    for v in cfg:
        if v is 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
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
            nn.Linear(512, 100),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        #print(' parameter1 = ', x.size())
        x = x.view(x.size(0), -1)
        #print(' parameter2 = ', x.size())
        x = self.classifier(x)
        return x


class CNNModelDeep(nn.Module):
    def __init__(self, features):
        super(CNNModelDeep, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            # ====================
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            # ====================
            nn.Linear(512, 100),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        #print(' parameter1 = ', x.size())
        x = x.view(x.size(0), -1)
        #print(' parameter2 = ', x.size())
        x = self.classifier(x)
        return x


class CNNModelWide(nn.Module):
    def __init__(self, features):
        super(CNNModelWide, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            # ====================
            nn.Linear(1024, 100),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        # print(' parameter1 = ', x.size())
        x = x.view(x.size(0), -1)
        # print(' parameter2 = ', x.size())
        x = self.classifier(x)
        return x



class CNNModelWide1(nn.Module):
    def __init__(self, features):
        super(CNNModelWide1, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            # ====================
            nn.Linear(1024, 100),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        # print(' parameter1 = ', x.size())
        x = x.view(x.size(0), -1)
        # print(' parameter2 = ', x.size())
        x = self.classifier(x)
        return x

class CNNModelWide2(nn.Module):
    def __init__(self, features):
        super(CNNModelWide2, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            # ====================
            nn.Linear(1024, 100),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        # print(' parameter1 = ', x.size())
        x = x.view(x.size(0), -1)
        # print(' parameter2 = ', x.size())
        x = self.classifier(x)
        return x

class CNNModelWide3(nn.Module):
    def __init__(self, features):
        super(CNNModelWide3, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 512),
            nn.ReLU(True),
            # ====================
            nn.Linear(512, 100),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        # print(' parameter1 = ', x.size())
        x = x.view(x.size(0), -1)
        # print(' parameter2 = ', x.size())
        x = self.classifier(x)
        return x

class CNNModelWide4(nn.Module):
    def __init__(self, features):
        super(CNNModelWide4, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 512),
            nn.ReLU(True),
            # ====================
            nn.Linear(512, 100),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        # print(' parameter1 = ', x.size())
        x = x.view(x.size(0), -1)
        # print(' parameter2 = ', x.size())
        x = self.classifier(x)
        return x


class CNNModelWide5(nn.Module):
    def __init__(self, features):
        super(CNNModelWide5, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 8192),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(8192, 512),
            nn.ReLU(True),
            # ====================
            nn.Linear(512, 100),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        # print(' parameter1 = ', x.size())
        x = x.view(x.size(0), -1)
        # print(' parameter2 = ', x.size())
        x = self.classifier(x)
        return x

class CNNModelWide6(nn.Module):
    def __init__(self, features):
        super(CNNModelWide6, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 16384),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(16384, 512),
            nn.ReLU(True),
            # ====================
            nn.Linear(512, 100),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        # print(' parameter1 = ', x.size())
        x = x.view(x.size(0), -1)
        # print(' parameter2 = ', x.size())
        x = self.classifier(x)
        return x

class CNNModelWide7(nn.Module):
    def __init__(self, features):
        super(CNNModelWide7, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 8192),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(8192, 1024),
            nn.ReLU(True),
            # ====================
            nn.Linear(1024, 100),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        # print(' parameter1 = ', x.size())
        x = x.view(x.size(0), -1)
        # print(' parameter2 = ', x.size())
        x = self.classifier(x)
        return x

class CNNModelWide8(nn.Module):
    def __init__(self, features):
        super(CNNModelWide8, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 16384),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(16384, 1024),
            nn.ReLU(True),
            # ====================
            nn.Linear(1024, 100),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        # print(' parameter1 = ', x.size())
        x = x.view(x.size(0), -1)
        # print(' parameter2 = ', x.size())
        x = self.classifier(x)
        return x

class CNNModelST(nn.Module):
    def __init__(self, features):
        super(CNNModelST, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            # ====================
            nn.Linear(4096, 100),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        # print(' parameter1 = ', x.size())
        x = x.view(x.size(0), -1)
        # print(' parameter2 = ', x.size())
        x = self.classifier(x)
        return x


# for cifar100
"""
def vgg11():
    return CNNModel(make_layers(cfg['A']))

def vgg11_bn():
    return CNNModel(make_layers(cfg['A'], batch_norm=True))

def vgg16():
    return CNNModel(make_layers(cfg['D']))

def vgg16_bn():
    return CNNModel(make_layers(cfg['D'], batch_norm=True))
"""

# for cifar100
def vgg16_bn():
    return CNNModelDeep(make_layers(cfg['D'], batch_norm=True))

def vgg16_bn_deep():
    return CNNModelDeep(make_layers(cfg['D'], batch_norm=True))

def vgg16_bn_wide():
    return CNNModelWide(make_layers(cfg['D'], batch_norm=True))

def vgg16_bn_wide1():
    return CNNModelWide1(make_layers(cfg['D'], batch_norm=True))

def vgg16_bn_wide2():
    return CNNModelWide2(make_layers(cfg['D'], batch_norm=True))

def vgg16_bn_wide3():
    return CNNModelWide3(make_layers(cfg['D'], batch_norm=True))

def vgg16_bn_wide4():
    return CNNModelWide4(make_layers(cfg['D'], batch_norm=True))

def vgg16_bn_wide5():
    return CNNModelWide5(make_layers(cfg['D'], batch_norm=True))

def vgg16_bn_wide6():
    return CNNModelWide6(make_layers(cfg['D'], batch_norm=True))

def vgg16_bn_wide7():
    return CNNModelWide7(make_layers(cfg['D'], batch_norm=True))

def vgg16_bn_wide8():
    return CNNModelWide8(make_layers(cfg['D'], batch_norm=True))


def vgg19_bn_wide5():
    return CNNModelWide5(make_layers(cfg['E'], batch_norm=True))



def vgg16_bn_st():
    return CNNModelST(make_layers(cfg['D'], batch_norm=True))