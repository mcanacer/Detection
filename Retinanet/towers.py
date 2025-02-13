import torch
import torch.nn as nn


class StochasticDepth(nn.Module):
    def __init__(self, survival_prob):
        super(StochasticDepth, self).__init__()
        self._survival_prob = survival_prob

    def forward(self, inputs):
        if self._survival_prob is None or self._survival_prob == 0 or not self.training: # changed
            return inputs

        shape = [inputs.shape[0]] + [1] * (inputs.ndim - 1)
        noise = torch.empty(shape, dtype=inputs.dtype)
        noise = noise.bernoulli(self._survival_prob)

        return torch.div(inputs, self._survival_prob) * noise


class ConvolutionalTower(nn.Module):
    def __init__(self, num_repeats, filter_size):
        super(ConvolutionalTower, self).__init__()
        self._num_repeats = num_repeats
        self._filter_size = filter_size

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
            layers.append(nn.BatchNorm2d(self._filter_size))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, inputs):
        outputs = {}
        for idx, inputs in inputs.items():
            outputs[idx] = self._layers(inputs)
        return outputs



