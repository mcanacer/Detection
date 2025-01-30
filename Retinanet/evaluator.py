import numpy as np
from coco import COCO


class COCOEvaluator(object):
    def __init__(self, categories=None, score_threshold=0.5):
        self._categories = categories
        self._score_threshold = score_threshold

        self._coco = COCO(self._categories)

    def clone(self):
        return COCOEvaluator(self._categories, self._score_threshold)


    def add(self, gt, pred):
        '''
        Args:
            gt: {
                image: [HxWxC],
                boxes: [Nx4] / normlized,
                labels: [N],
            }
            pred: {
                boxes: [Mx4] / normalized,
                scores: [M],
                labels: [M],
            }

        '''

        height, width = 512, 512
        box_scaler = np.array([height, width, height, width], dtype=np.float32)

        gt_indices = gt[3] > 0.0
        pred_indices = pred[1] > self._score_threshold

        self._coco.add(
            gt_boxes=gt[1][gt_indices] *
            box_scaler,
            gt_labels=gt[2][gt_indices],
            pred_boxes=pred[0][pred_indices] *
            box_scaler,
            pred_labels=pred[2][pred_indices],
            pred_scores=pred[1][pred_indices],
            height=height,
            width=width,
        )

    def evaluate(self):
        return self._coco.evaluate()


