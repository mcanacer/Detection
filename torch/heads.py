import torch.nn as nn


class ConvolutionalHead(nn.Module):
    def __init__(self, num_repeats, filter_size, out_filters):
        super(ConvolutionalHead, self).__init__()
        self._num_repeats = num_repeats
        self._filter_size = filter_size
        self._out_filters = out_filters

        self._layers = self._make_layers()

    def _make_layers(self):
        layers = []
        for _ in range(self._num_repeats):
            layers.append(nn.Conv2d(self._filter_size, self._filter_size, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(self._filter_size, self._out_filters, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self._layers(x)

