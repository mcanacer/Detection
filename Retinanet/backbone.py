import torch.nn as nn

from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision import models
from torchvision.models import feature_extraction


class ResNet(nn.Module):
    def __init__(self, backbone="resnet18", weights="IMAGENET1K_V1"):
        super(ResNet, self).__init__()
        if backbone == 'resnet18':
            backbone = resnet18(weights=weights)
            self.out_channels = [128, 256, 512]
        elif backbone == 'resnet34':
            backbone = resnet34(weights=weights)
            self.out_channels = [128, 256, 512]
        elif backbone == 'resnet50':
            backbone = resnet50(weights=weights)
            self.out_channels = [512, 1024, 2048]
        elif backbone == 'resnet101':
            backbone = resnet101(weights=weights)
            self.out_channels = [512, 1024, 2048]
        else:  # backbone == 'resnet152':
            backbone = resnet152(weights=weights)
            self.out_channels = [512, 1024, 2048]
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:5])

        self.c3 = list(backbone.children())[5]
        self.c4 = list(backbone.children())[6]
        self.c5 = list(backbone.children())[7]

    def forward(self, x):
        x = self.feature_extractor(x)
        c3 = self.c3[0](x)
        c4 = self.c4[0](c3)
        c5 = self.c5[0](c4)
        return c3, c4, c5

class RegNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize with ImageNet pre-trained weights.
        _cnn = models.regnet_x_400mf()

        # Torchvision models only return features from the last level. Detector
        # backbones (with FPN) require intermediate features of different scales.
        # So we wrap the ConvNet with torchvision's feature extractor. Here we
        # will get output features with names (c3, c4, c5) with same stride as
        # (p3, p4, p5) described above.
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            },
        )

        self.out_channels = [64, 160, 400]

    def forward(self, x):
        feats = self.backbone(x)
        return feats['c3'], feats['c4'], feats['c5']


class MobileNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetBackbone, self).__init__()
        mobilenet = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
        self.features = mobilenet.features

        self.c3 = nn.Sequential(*self.features[:6])  # 1/16 scale
        self.c4 = nn.Sequential(*self.features[6:12])  # 1/32 scale
        self.c5 = nn.Sequential(*self.features[12:])  # 1/64 scale

        self.out_channels = [40, 112, 960]

    def forward(self, x):
        c3 = self.c3(x)
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        return c3, c4, c5


def ResNet18():
    return ResNet()

def ResNet34():
    return ResNet(backbone="resnet34")

def ResNet50():
    return ResNet(backbone="resnet50")

def ResNet101():
    return ResNet(backbone="resnet101")

def ResNet152():
    return ResNet(backbone="resnet152")


