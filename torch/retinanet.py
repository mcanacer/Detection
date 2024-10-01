import losses
import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):

    def __init__(
            self,
            backbone,
            fpn,
            box_head,
            class_head,
            feature_map_indexes,
    ):
        super(FeatureExtractor, self).__init__()
        self._backbone = backbone
        self._fpn = fpn
        self._box_head = box_head
        self._class_head = class_head
        self._feature_map_indexes = feature_map_indexes

    def forward(self, inputs):
        inputs = self._backbone(inputs)
        inputs = self._fpn(inputs)

        inputs = {str(idx): inputs[idx] for idx in self._feature_map_indexes}

        box_preds = inputs
        box_preds = {
            str(idx): box_pred[idx]
            for idx, box_pred in box_preds.items()
        }

        class_preds = inputs
        class_preds = {
            str(idx): class_pred[idx]
            for idx, class_pred in class_preds.items()
        }

        box_preds = [box_preds[idx] for idx in sorted(box_preds.keys())]
        class_preds = [class_preds[idx] for idx in sorted(class_preds.keys())]

        strides = [2 ** idx for idx in sorted(self._feature_map_indexes)]
        return box_preds, class_preds, strides


class RetinaNet(nn.Module):

    def __init__(
            self,
            model,
            num_classes,
            anchor_generator,
            assigner,
            box_coder,
            max_detections=40,
            iou_threshold=0.5,
            class_losses=[('focal', 1.0, losses.Focal())],
            regression_losses=[('l1', 1.0, losses.L1())],
    ):
        super(RetinaNet, self).__init__()
        self._model = model

        self._num_classes = num_classes
        self._anchor_generator = anchor_generator
        self._assigner = assigner
        self._box_coder = box_coder

        self._max_detections = max_detections
        self._iou_threshold = iou_threshold

        self._classes_losses = class_losses
        self._regression_losses = regression_losses

    def forward(self, inputs, mode='losses'):
        return getattr(self, mode)(inputs)

    def losses(self, inputs):
        images = inputs[0]  # [N, 3, W, H]
        gt_boxes = inputs[1]  # [N, T, 4]
        gt_labels = inputs[2]  # [N, T]

        # box_preds: [N, W, H, Ax4], class_preds: [N, W, H, AxC]
        box_preds, class_preds, strides = self._model(images)

        feature_map_sizes = [
            box_pred.shape[1:3] for box_pred in box_preds
        ]

        box_preds = cat(box_preds, (-1, 4), 1)  # [N, M, 4]
        class_preds = cat(class_preds, (-1, self._num_classes), 1)  # [N, M, C]

        anchors = self._anchor_generator(feature_map_sizes, strides)  # [M, 4]

        (
            target_boxes,  # [N, M, 4]
            target_boxes_mask,  # [N, M]
            target_labels,  # [N, M]
            target_labels_mask,  # [N, M]
            num_matches,  # [N]
        ) = self._assigner(
            anchors,  # [M, 4]
            gt_boxes,  # [N, T, 4]
            gt_labels,  # [N, T]
        )

        target_labels = torch.one_hot(target_labels, self._num_classes)


def cat(inputs_list, shape, dim):
    return torch.cat(
        [inputs.contiguous().view(inputs.shape[0], shape) for inputs in inputs_list],
        dim=dim,
    )

