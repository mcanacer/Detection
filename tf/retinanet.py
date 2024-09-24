import tensorflow as tf

from . import losses


class FeatureExtractor(tf.keras.layers.Layer):

    def __init__(
            self,
            backbone,
            fpn,
            box_head,
            cls_head,
            feature_map_indexes,
            **kwargs,
    ):
        super(FeatureExtractor, self).__init__(**kwargs)

        self._backbone = backbone
        self._fpn = fpn
        self._box_head = box_head
        self._cls_head = cls_head

        self._feature_map_indexes = feature_map_indexes

    def __call__(self, inputs, training=None):
        inputs = self._backbone(inputs, training=training)
        inputs = self._fpn(inputs, training=training)

        inputs = {str(idx): inputs[idx] for idx in self._feature_map_indexes}

        box_preds = inputs
        box_preds = {
            idx: self._box_head(box_pred, training=training)
            for idx, box_pred in box_preds.items()
        }

        cls_preds = inputs
        cls_preds = {
            idx: self._cls_head(cls_pred)
            for idx, cls_pred in cls_preds.items()
        }

        box_preds = [box_preds[idx] for idx in sorted(box_preds.keys())]
        cls_preds = [cls_preds[idx] for idx in sorted(cls_preds.keys())]

        strides = [2 ** idx for idx in sorted(self._feature_map_indexes)]
        return box_preds, cls_preds, strides


class RetinaNet(object):

    def __init__(
            self,
            model,
            num_classes,
            anchor_generator,
            assigner,
            box_coder,
            max_detections=40,
            iou_threshold=0.5,
            score_threshold=0.0,
            regression_losses=[('huber', 1.0, losses.Huber())],
            cls_losses=[('focal', 1.0, losses.Focal())],
    ):
        self._model = model

        self._num_classes = num_classes

        self._anchor_generator = anchor_generator
        self._assigner = assigner
        self._box_coder = box_coder

        self._max_detections = max_detections
        self._iou_threshold = iou_threshold
        self._score_threshold = score_threshold

        self._regression_losses = regression_losses
        self._cls_losses = cls_losses

    def losses(self, inputs, training):
        pass



