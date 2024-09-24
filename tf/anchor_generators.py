import tensorflow as tf
import collections


class GridAnchorGenerator(object):

    def __init__(self, scales, aspect_ratios, scale=1.0, max_clip_value=0.99):
        self._scales = list(map(lambda x: x * scale, scales))
        self._aspect_ratios = aspect_ratios
        self._max_clip_value = max_clip_value

        self._num = len(self._scales) * len(self._aspect_ratios)

    def num_anchors_per_location(self):
        return self._num

    def __call__(self, height, width):
        meshgrid = lambda x, y: [
            tf.reshape(output, -1) for output in tf.meshgrid(x, y)
        ]

        scales = tf.constant(self._scales, dtype=tf.float32)
        aspect_ratios = tf.constant(self._aspect_ratios, dtype=tf.float32)
        scales, aspect_ratios = meshgrid(scales, aspect_ratios)

        aspect_ratios_sqrts = tf.math.sqrt(aspect_ratios)
        widths = scales / aspect_ratios_sqrts
        heights = scales * aspect_ratios_sqrts

        x_centers, y_centers = meshgrid(
            tf.range(width, dtype=tf.float32),
            tf.range(height, dtype=tf.float32)
        )
        heights, y_centers = meshgrid(heights, y_centers)
        widths, x_centers = meshgrid(widths, x_centers)

        f_height = tf.cast(height, tf.float32)
        f_width = tf.cast(width, tf.float32)

        xmin = (x_centers - widths) / f_width
        ymin = (y_centers - heights) / f_height
        xmax = (x_centers + widths) / f_width
        ymax = (y_centers + heights) / f_height

        boxes = tf.stack([xmin, ymin, xmax, ymax], axis=-1)
        boxes = tf.clip_by_value(boxes, 0.0, self._max_clip_value)
        return boxes


Anchors = collections.namedtuple(
    'Anchors', ['boxes', 'sizes', 'strides', 'nums', 'indices', 'total'])


class MultipleGridAnchorGenerator(object):

    def __init__(self, scales, aspect_ratios, scale=1.0, max_clip_value=0.99):
        self._generator = GridAnchorGenerator(
            scales,
            aspect_ratios,
            scale=scale,
            max_clip_value=max_clip_value
        )

    def __call__(self, sizes, strides):
        anchors, indices = [], []

        start = 0
        for height, width in sizes:
            level_anchors = self._generator(height, width)
            num, _ = level_anchors.shape.as_list()

            anchors.append(level_anchors)
            indices.append((start, start + num))

            start += num

        return Anchors(
            boxes=tf.concat(anchors, axis=0),
            sizes=sizes,
            strides=strides,
            nums=[self._generator.num_anchors_per_location()] * len(sizes),
            indices=indices,
            total=indices[-1][-1],
        )
