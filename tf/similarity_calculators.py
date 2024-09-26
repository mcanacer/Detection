import tensorflow as tf

from . import boxops


class IouSimilarity(object):

    def __call__(self, boxes1, boxes2, batch_dims=0):
        return boxops.iou(boxes1, boxes2, batch_dims)
