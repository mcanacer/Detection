import torch
import torch.nn as nn
import numpy as np
import collections


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

        xmin = (x_centers - widths) / width
        ymin = (y_centers - heights) / height
        xmax = (x_centers + widths) / width
        ymax = (y_centers + heights) / height

        boxes = torch.stack([xmin, ymin, xmax, ymax], dim=-1)
        boxes = torch.clip(boxes, min=0.0, max=self._max_clip_value)
        return boxes

Anchors = collections.namedtuple(
    'Anchors', ['boxes', 'sizes', 'strides', 'nums', 'indices', 'total']
)


class MultipleGridAnchor(object):

    def __init__(self, scales, aspect_ratios, scale=1.0, max_clip_value=0.99):
        self._generators = GridAnchorGenerator(
            scales,
            aspect_ratios,
            scale,
            max_clip_value
        )

    def __call__(self, sizes, strides):
        anchors, indices = [], []

        start = 0
        for height, width in sizes:
            level_anchors = self._generators(height, width)
            num, _ = level_anchors.shape

            anchors.append(level_anchors)
            indices.append((start, start + num))

            start += num

        return Anchors(
            boxes=torch.cat(anchors, dim=0),
            sizes=sizes,
            strides=strides,
            nums=[self._generators.num_anchors_per_location()] * len(sizes),
            indices=indices,
            total=indices[-1][-1]
        )


