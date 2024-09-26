import tensorflow as tf


def area(boxes):
    xmin, ymin, xmax, ymax = tf.unstack(boxes, axis=-1)
    return (xmax - xmin) * (ymax - ymin)


def intersection(boxes1, boxes2, batch_dims=0):
    boxes1_ndims = boxes1.shape.rank
    boxes2_ndims = boxes2.shape.rank

    for _ in range(boxes2_ndims - batch_dims - 1):
        boxes1 = tf.expand_dims(boxes1, axis=-2)

    for _ in range(boxes1_ndims - batch_dims - 1):
        boxes2 = tf.expand_dims(boxes2, axis=batch_dims)

    xmin_1, ymin_1, xmax_1, ymax_1 = tf.unstack(boxes1, axis=-1)
    xmin_2, ymin_2, xmax_2, ymax_2 = tf.unstack(boxes2, axis=-1)

    xmin = tf.maximum(xmin_1, xmin_2)
    ymin = tf.maximum(ymin_1, ymin_2)
    xmax = tf.maximum(xmax_1, xmax_2)
    ymax = tf.maximum(ymax_1, ymax_2)

    return tf.maximum(0.0, xmax - xmin) * tf.maximum(0.0, ymax - ymin)


def iou(boxes1, boxes2, batch_dims=0):
    boxes1_ndims = boxes1.shape.rank
    boxes2_ndims = boxes2.shape.rank

    for _ in range(boxes2_ndims - batch_dims - 1):
        boxes1 = tf.expand_dims(boxes1, axis=-2)

    for _ in range(boxes1_ndims - batch_dims - 1):
        boxes2 = tf.expand_dims(boxes2, axis=batch_dims)

    xmin_1, ymin_1, xmax_1, ymax_1 = tf.unstack(boxes1, axis=-1)
    xmin_2, ymin_2, xmax_2, ymax_2 = tf.unstack(boxes2, axis=-1)

    height_1 = tf.math.maximum(0.0, ymax_1 - ymin_1)
    width_1 = tf.math.maximum(0.0, xmax_1 - xmin_1)

    height_2 = tf.math.maximum(0.0, ymax_2 - ymin_2)
    width_2 = tf.math.maximum(0.0, xmax_2 - xmin_2)

    area_1 = height_1 * width_1
    area_2 = height_2 * width_2

    ymin_intersection = tf.math.maximum(ymin_1, ymin_2)
    xmin_intersection = tf.math.maximum(xmin_1, xmin_2)
    ymax_intersection = tf.math.minimum(ymax_1, ymax_2)
    xmax_intersection = tf.math.minimum(xmax_1, xmax_2)

    height_intersection = tf.math.maximum(
        0.0,
        ymax_intersection - ymin_intersection,
    )
    width_intersection = tf.math.maximum(
        0.0,
        xmax_intersection - xmin_intersection,
    )

    intersections = height_intersection * width_intersection

    unions = area_1 + area_2 - intersections
    return tf.math.divide_no_nan(intersections, unions)

