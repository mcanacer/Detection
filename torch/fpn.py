import torch.nn as nn


class FPN(nn.Module):

    def __init__(self, out_channels, filter_size):
        super(FPN, self).__init__()
        c3_size, c4_size, c5_size = out_channels
        self.p3_in = nn.Conv2d(c3_size, filter_size, kernel_size=1)
        self.p4_in = nn.Conv2d(c4_size, filter_size, kernel_size=1)
        self.p5_in = nn.Conv2d(c5_size, filter_size, kernel_size=1)

        self.p3_out = nn.Conv2d(filter_size, filter_size, kernel_size=3, stride=1, padding=1)
        self.p4_out = nn.Conv2d(filter_size, filter_size, kernel_size=3, stride=1, padding=1)
        self.p5_out = nn.Conv2d(filter_size, filter_size, kernel_size=3, stride=1, padding=1)

        self.p6_out = nn.Conv2d(c5_size, filter_size, kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU()
        self.p7_out = nn.Conv2d(filter_size, filter_size, kernel_size=3, stride=2, padding=1)

    def _upsampled_add(self, x, y):
        _, _, H, W = y.size()
        return nn.functional.interpolate(x, size=(H, W), mode='nearest') + y

    def forward(self, inputs):
        c3, c4, c5 = inputs

        p5_in = self.p5_in(c5)
        p4_in = self.p4_in(c4)
        p3_in = self.p3_in(c3)

        p5_out = self.p5_out(p5_in)
        p4_out = self.p4_out(self._upsampled_add(p5_in, p4_in))
        p3_out = self.p3_out(self._upsampled_add(p3_in, self._upsampled_add(p5_in, p4_in)))
        p6_out = self.p6_out(c5)
        p7_out = self.p7_out(self.relu(p6_out))

        return p3_out, p4_out, p5_out, p6_out, p7_out


