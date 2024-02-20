import torch
from torch import nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    def __init__(self, channel_in, channel_out, is_downsample=False):
        super(BasicBlock, self).__init__()

        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(channel_in, channel_out, 3, 2, 1, bias=False)
        else:
            self.conv1 = nn.Conv2d(channel_in, channel_out, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel_out, channel_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel_out)

        if is_downsample:
            self.down_sample = nn.Sequential(
                nn.Conv2d(channel_in, channel_out, 1, 2, 0, bias=False),
                nn.BatchNorm2d(channel_out),
            )
        elif channel_in != channel_out:
            self.down_sample = nn.Sequential(
                nn.Conv2d(channel_in, channel_out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(channel_out),
            )
            self.is_downsample = True

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.down_sample(x)
        return F.relu(x.add(y), True)


def make_layers(channel_in, channel_out, repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks += [BasicBlock(channel_in, channel_out, is_downsample=is_downsample)]
        else:
            blocks += [BasicBlock(channel_out, channel_out)]
    return nn.Sequential(*blocks)


class Net(nn.Module):
    def __init__(self, num_classes=751, reid=False):
        super(Net, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )

        # Make layer
        self.layer1 = make_layers(64, 64, 2, False)
        self.layer2 = make_layers(64, 128, 2, True)
        self.layer3 = make_layers(128, 256, 2, True)
        self.layer4 = make_layers(256, 512, 2, True)

        # Average pooling
        self.avgpool = nn.AvgPool2d((8, 4), 1)

        # Reid
        self.reid = reid

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # B x 128
        if self.reid:
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            return x

        # Classifier
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    net = Net()
    x = torch.randn(4, 3, 128, 64)
    y = net(x)
    import ipdb

    ipdb.set_trace()
