import torch.nn as nn
import torch.nn.functional as F
import math

# dymatic creating model

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
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
            nn.Linear(512, 10),
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




def vgg11():
    return CNNModel(make_layers(cfg['A']))

def vgg11_bn():
    return CNNModel(make_layers(cfg['A'], batch_norm=True))


