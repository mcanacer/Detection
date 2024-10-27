import torch
import torch.nn as nn


class StochasticDepth(nn.Module):
    def __init__(self, survival_prob):
        super(StochasticDepth, self).__init__()
        self._survival_prob = survival_prob

    def forward(self, inputs, training=None):
        if self._survival_prob is None or self._survival_prob == 0 or not training:
            return inputs

        shape = [inputs.shape[0]] + [1] * (inputs.ndim - 1)
        noise = torch.empty(shape, dtype=inputs.dtype)
        noise = noise.bernoulli(self._survival_prob)

        return torch.div(inputs, self._survival_prob) * noise


class ConvolutionalTower(nn.Module):
    def __init__(self, num_repeats, filter_size, survival_prob, use_residual=True):
        super(ConvolutionalTower, self).__init__()
        self._num_repeats = num_repeats
        self._filter_size = filter_size
        self._survival_prob = survival_prob
        self._use_residual = use_residual

        self._stochastic_depth = StochasticDepth(survival_prob)

        self._make_layers()

        self.relu = nn.ReLU()

    def _make_layers(self):
        self._conv_layers = []
        self._bn_layers = []
        for _ in range(self._num_repeats):
            self._conv_layers.append(nn.Conv2d(self._filter_size, self._filter_size, kernel_size=3, stride=1, padding=1))
            self._bn_layers.append(nn.BatchNorm2d(self._filter_size))

    def forward(self, inputs, training=None):
        def conv_norm_act(conv, bn, inputs, use_residual):
            outputs = self.relu(bn(conv(inputs)))
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
        return inputs



