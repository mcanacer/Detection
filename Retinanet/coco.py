import datetime
import numpy as np
import copy

def get_center_coordinates_and_sizes(boxes):
    ymin, xmin, ymax, xmax = [
        np.squeeze(x, -1) for x in np.split(boxes, 4, -1)
    ]

    width = xmax - xmin
    height = ymax - ymin

    ycenter = ymin + height / 2.
    xcenter = xmin + width / 2.

    return [ycenter, xcenter, height, width]


def compute_area(boxes):
    ymin, xmin, ymax, xmax = [
        np.squeeze(x, -1) for x in np.split(boxes, 4, -1)
    ]
    return (ymax - ymin) * (xmax - xmin)


class COCO(object):

    def __init__(self, categories=None):
        now = datetime.datetime.now()
        now = now.strftime('%I:%M %p on %B %d, %Y')

        self._gt = {
            'info': {
                'description': 'COCO EVAL',
                'version': '1.0.0',
                'contributor': 'Muhammet Can ACER',
                'date_created': now,
            },
            'licenses': [{
                'id': 1,
                'name': 'Default License',
                'url': ''
            }],
            'images': [],
            'annotations': [],
            'categories': categories,
        }

        self._dt = {
            'annotations': [],
        }

        self._image_id = 1
        self._gt_id = 1
        self._dt_id = 1

        self._categories = set()

        self._metric_names = [
            'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'ARmax1', 'ARmax10',
            'ARmax100', 'ARs', 'ARm', 'ARl'
        ]


    def evaluate(self):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        categories = [{'id': idx} for idx in self._categories]
        categories = self._gt['categories'] or categories

        gt = copy.deepcopy(self._gt)
        gt['categories'] = copy.deepcopy(categories)

        dt = copy.deepcopy(self._dt)
        dt['images'] = copy.deepcopy(self._gt['images'])
        dt['categories'] = copy.deepcopy(categories)

        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()

        coco_dt = COCO()
        coco_dt.dataset = dt
        coco_dt.createIndex()

        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        metrics = coco_eval.stats
        return {name: metrics[i] for i, name in enumerate(self._metric_names)}


    def add(
        self,
        gt_boxes,
        gt_labels,
        pred_boxes,
        pred_labels,
        pred_scores,
        height,
        width,
    ):
        '''
            Args:
                gt_boxes shape: Nx4, [ymin, xmin, ymax, xmax] / unnormalized
                gt_labels: shape: N

                pred_boxes shape: Nx4, [ymin, xmin, ymax, xmax], unnormalized
                pred_labels: shape: N
                pred_scores: shae: N
        '''

        ymin, xmin, ymax, xmax = [
            np.squeeze(x, -1) for x in np.split(pred_boxes, 4, -1)
        ]
        area = (ymax - ymin) * (xmax - xmin)
        for label, score, area, xmin, ymin, xmax, ymax in zip(
                pred_labels.tolist(),
                pred_scores.tolist(),
                area.tolist(),
                xmin.tolist(),
                ymin.tolist(),
                xmax.tolist(),
                ymax.tolist(),
        ):
            self._dt['annotations'].append({
                'image_id':
                self._image_id,
                'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                'score':
                float(score),
                'category_id':
                int(label),
                'area':
                area,
                'iscrowd':
                0,
                'id':
                self._dt_id,
            })
            self._dt_id += 1

        ymin, xmin, ymax, xmax = [
            np.squeeze(x, -1) for x in np.split(gt_boxes, 4, -1)
        ]
        area = (ymax - ymin) * (xmax - xmin)
        for label, area, xmin, ymin, xmax, ymax in zip(
                gt_labels.tolist(),
                area.tolist(),
                xmin.tolist(),
                ymin.tolist(),
                xmax.tolist(),
                ymax.tolist(),
        ):
            self._gt['annotations'].append({
                'id':
                self._gt_id,
                'image_id':
                self._image_id,
                'category_id':
                label,
                'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                'area':
                area,
                'iscrowd':
                0,
            })
            self._categories.add(label)

            self._gt_id += 1

        self._gt['images'].append({
            'id': self._image_id,
            'width': width,
            'height': height,
            'license': 1,
        })
        self._image_id += 1
