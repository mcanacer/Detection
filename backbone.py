import torch.nn as nn

from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152


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

