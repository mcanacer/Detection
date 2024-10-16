import torch
import torch.nn as nn
import numpy as np


class GridAnchorGenerator(object):

    def __init__(self, scales, aspect_ratios, scale=1.0, max_clip_value=0.99):
        self._scales = list(map(lambda x: x * scale, scales))
        self._aspect_ratios = aspect_ratios
        self._max_clip_value = max_clip_value

        self._num = len(self._scales) * len(self._aspect_ratios)

    def num_anchors_per_location(self):
        return self._num

    def __call__(self, width, height):
        meshgrid = lambda x, y: [
            output.reshape(-1) for output in torch.meshgrid(x, y, indexing='xy')
        ]

        scales = torch.tensor(self._scales, dtype=torch.float32)
        aspect_ratios = torch.tensor(self._aspect_ratios, dtype=torch.float32)
        scales, aspect_ratios = meshgrid(scales, aspect_ratios)

        aspect_ratios_sqrt = torch.sqrt(aspect_ratios)
        widths = scales * aspect_ratios_sqrt
        heights = scales / aspect_ratios_sqrt

        x_centers, y_centers = meshgrid(
            torch.arange(width, dtype=torch.float32),
            torch.arange(height, dtype=torch.float32)
        )

        heights, y_centers = meshgrid(heights, y_centers)
        widths, x_centers = meshgrid(widths, x_centers)

        xmin = (x_centers - 0.5 * widths) / width
        ymin = (y_centers - 0.5 * heights) / height
        xmax = (x_centers + 0.5 * widths) / width
        ymax = (y_centers + 0.5 * heights) / height

        boxes = torch.stack([xmin, ymin, xmax, ymax], dim=-1)
        boxes = torch.clip(boxes, min=0.0, max=self._max_clip_value)
        return boxes


class MultipleGridAnchor(object):

    def __init__(self, scales, aspect_ratios, scale=1.0, max_clip_value=0.99):
        self._generators = GridAnchorGenerator(
            scales,
            aspect_ratios,
            scale,
            max_clip_value
        )

    def __call__(self, sizes, strides):
        anchors = []

        for height, width in sizes:
            level_anchors = self._generators(height, width)
            anchors.append(level_anchors)

        return torch.cat(anchors, dim=0)


class Anchors(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super().__init__()
        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        if strides is None:
            self.strides = [2 ** s for s in self.pyramid_levels]
        if sizes is None:
            self.sizes = [2 ** (s + 2) for s in self.pyramid_levels]
        if ratios is None:
            self.ratios = [0.5, 1, 2]
        if scales is None:
            self.scales = [2 ** (i/3) for i in range(3)]

    def forward(self, images):
        image_shape = images.shape[-2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** i - 1) // 2 ** i for i in self.pyramid_levels]

        all_anchors = np.zeros((0, 4)).astype(np.float32)

        for idx, p in enumerate(self.pyramid_levels):
            anchors = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

        all_anchors = np.expand_dims(all_anchors, axis=0)

        anchors = torch.from_numpy(all_anchors.astype(np.float32))
        if torch.cuda.is_available():
            anchors = anchors.cuda()
        return anchors


def generate_anchors(base_size, ratios, scales):
    num_anchors = len(ratios) * len(scales)

    anchors = np.zeros((num_anchors, 4))

    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    areas = anchors[:, 2] * anchors[:, 3]

    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[0]) + 0.5) * stride
    shift_y = (np.arange(0, shape[1]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K*A, 4))

    return all_anchors
