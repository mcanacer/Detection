import torch
import torch.nn as nn
from towers import StochasticDepth


class ConvolutionalHead(nn.Module):
    def __init__(self, num_repeats, filter_size, out_filters, survival_prob=0.5, use_residual=True):
        super(ConvolutionalHead, self).__init__()
        self._num_repeats = num_repeats
        self._filter_size = filter_size
        self._out_filters = out_filters
        self._survival_prob = survival_prob
        self._use_residual = use_residual

        self._layers = self._make_layers()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layers(self):
        layers = []
        for i in range(self._num_repeats):
            layers.append(nn.Conv2d(self._filter_size, self._filter_size, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
        self._out_conv = nn.Conv2d(self._filter_size, self._out_filters, kernel_size=3, stride=1, padding=1)
        return nn.Sequential(*layers)


    def forward(self, inputs):
        return self._out_conv(self._layers(inputs))




