import torch.nn as nn
from towers import StochasticDepth


class ConvolutionalHead(nn.Module):
    def __init__(self, num_repeats, filter_size, out_filters, survival_prob, use_residual=True):
        super(ConvolutionalHead, self).__init__()
        self._num_repeats = num_repeats
        self._filter_size = filter_size
        self._out_filters = out_filters

        self._stochastic_depth = StochasticDepth(survival_prob)

        self._use_residual = use_residual

        self._make_layers()

        self.relu = nn.ReLU()

    def _make_layers(self):
        self._conv_layers = []
        self._bn_layers = []
        for _ in range(self._num_repeats):
            self._conv_layers.append(nn.Conv2d(self._filter_size, self._filter_size, kernel_size=3, stride=1, padding=1))
            self._bn_layers.append(nn.BatchNorm2d(self._filter_size))
        self._out_conv = nn.Conv2d(self._filter_size, self._out_filters, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs, training=None):
        def conv_norm_act(conv, norm, inputs, use_residual):
            outputs = self.relu(norm(conv(inputs)))
            if use_residual:
                outputs = self._stochastic_depth(outputs, training=training)
                return inputs + outputs
            else:
                return outputs
        for i, (conv, bn) in enumerate(zip(self._conv_layers, self._bn_layers)):
            inputs = conv_norm_act(
                conv,
                bn,
                inputs,
                i > 0 and self._use_residual,
            )
        return self._out_conv(inputs)




