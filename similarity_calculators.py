from losses import calc_iou


class IouSimilarity(object):

    def __call__(self, boxes1, boxes2):
        return calc_iou(boxes1, boxes2)
