from boxops import iou


class IouSimilarity(object):

    def __call__(self, boxes1, boxes2):
        return iou(boxes1, boxes2)
