import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, out_channels, feature_size=256):
        super().__init__()
        self.feature_size = feature_size

        self.c3_to_p3 = nn.Conv2d(out_channels[-3], feature_size, kernel_size=1)
        self.c4_to_p4 = nn.Conv2d(out_channels[-2], feature_size, kernel_size=1)
        self.c5_to_p5 = nn.Conv2d(out_channels[-1], feature_size, kernel_size=1)
        self.c5_to_p6 = nn.Conv2d(out_channels[-1], feature_size, kernel_size=3, stride=2, padding=1)
        self.p6_to_p7 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        self.p7_out = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.p6_out = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.p5_out = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.p4_out = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.p3_out = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, inputs):
        c3, c4, c5 = inputs

        p3 = self.c3_to_p3(c3)
        p4 = self.c4_to_p4(c4)
        p5 = self.c5_to_p5(c5)
        p6 = self.c5_to_p6(c5)
        p7 = self.p6_to_p7(self.relu(p6))

        p7_out = self.p7_out(p7)
        p6 = self._upsample_add(p7, p6)
        p6_out = self.p6_out(p6)
        p5 = self._upsample_add(p6, p5)
        p5_out = self.p5_out(p5)
        p4 = self._upsample_add(p5, p4)
        p4_out = self.p4_out(p4)
        p3 = self._upsample_add(p4, p3)
        p3_out = self.p3_out(p3)

        return p3_out, p4_out, p5_out, p6_out, p7_out

