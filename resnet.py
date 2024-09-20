import torch.nn as nn

import math


class BasicBlock(nn.Module):

    expand_ratio = 1

    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super().__init__()
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class BottleNeck(nn.Module):

    expand_ratio = 4

    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super().__init__()
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expand_ratio, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expand_ratio)

    def forward(self, x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes, out_channels=None, backbone=True):
        super().__init__()
        if out_channels is None:
            out_channels = [64, 128, 256, 512]
        if backbone:
            self.out_channels = [c * block.expand_ratio for c in out_channels][-3:]
        self.in_channels = 64
        self.backbone = backbone
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layers(block, layers[0], out_channel=out_channels[0], stride=1)

        self.layer2 = self._make_layers(block, layers[1], out_channel=out_channels[1], stride=2)

        self.layer3 = self._make_layers(block, layers[2], out_channel=out_channels[2], stride=2)

        self.layer4 = self._make_layers(block, layers[3], out_channel=out_channels[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512*block.expand_ratio, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        x = self.avgpool(c5)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return (c3, c4, c5) if self.backbone else x

    def _make_layers(self, block, num_residual_blocks, out_channel, stride):
        downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channel*block.expand_ratio:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channel*block.expand_ratio, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel*block.expand_ratio)
            )

        layers.append(block(self.in_channels, out_channel, downsample=downsample, stride=stride))

        self.in_channels = out_channel*block.expand_ratio

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channel))

        return nn.Sequential(*layers)


def ResNet18(image_channels=3, num_classes=10, backbone=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], image_channels, num_classes, backbone=backbone)

def ResNet34(image_channels=3, num_classes=10, backbone=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], image_channels, num_classes, backbone=backbone)

def ResNet50(image_channels=3, num_classes=10, backbone=True):
    return ResNet(BottleNeck, [3, 4, 6, 3], image_channels, num_classes, backbone=backbone)

def ResNet101(image_channels=3, num_classes=10, backbone=True):
    return ResNet(BottleNeck, [3, 4, 23, 3], image_channels, num_classes, backbone=backbone)

def ResNet152(image_channels=3, num_classes=10, backbone=True):
    return ResNet(BottleNeck, [3, 8, 36, 3], image_channels, num_classes, backbone=backbone)


