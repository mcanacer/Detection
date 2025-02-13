import losses

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import math

class FeatureExtractor(nn.Module):

    def __init__(
            self,
            backbone,
            fpn,
            box_tower,
            class_tower,
            box_head,
            class_head,
            feature_map_indexes,
    ):
        super(FeatureExtractor, self).__init__()
        self._backbone = backbone
        self._fpn = fpn
        self._box_tower = box_tower
        self._class_tower = class_tower
        self._box_head = box_head
        self._class_head = class_head
        self._feature_map_indexes = feature_map_indexes

        prior = 0.01

        self._class_head._out_conv.bias.data.fill_(math.log(prior / (1.0 - prior)))


    def forward(self, inputs):
        inputs = self._backbone(inputs)
        inputs = self._fpn(inputs)

        inputs = {str(idx): inputs[idx] for idx in self._feature_map_indexes}

        box_preds = inputs

        box_preds = self._box_tower(box_preds)

        box_preds = {
            str(idx): self._box_head(box_pred)
            for idx, box_pred in box_preds.items()
        }

        class_preds = inputs

        class_preds = self._class_tower(class_preds)

        class_preds = {
            str(idx): self._class_head(class_pred)
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
            box_losses=[('Iou', 1.0, losses.IOULoss('ciou'))],
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
        self._box_losses = box_losses

    def forward(self, inputs, mode='losses'):
        return getattr(self, mode)(inputs)

    def losses(self, inputs):
        images = inputs[0]  # [N, 3, H, W]
        gt_boxes = inputs[1]  # [N, T, 4]
        gt_labels = inputs[2]  # [N, T]
        gt_weights = inputs[3]  # [N, T]

        # box_preds: [N, Ax4, H, W], class_preds: [N, AxC, H, W]
        box_preds, class_preds, strides = self._model(images)

        feature_map_sizes = [
            box_pred.shape[-2:] for box_pred in box_preds
        ]

        box_preds = cat(box_preds, (-1, 4), 1)  # [N, M, 4]
        class_preds = cat(class_preds, (-1, self._num_classes), 1)  # [N, M, C]

        anchors = self._anchor_generator(feature_map_sizes, strides)  # [M, 4]

        (
            target_boxes,  # [N, M, 4]
            target_boxes_weights,  # [N, M]
            target_labels,  # [N, M]
            target_labels_weights,  # [N, M]
            num_matches,  # [N]
        ) = self._assigner(
            anchors,  # [M, 4]
            gt_boxes,  # [N, T, 4]
            gt_labels,  # [N, T]
            gt_weights,  # [N, T]
        )

        target_labels = one_hot(target_labels - 1, self._num_classes)  # [N, M, C]
        target_labels *= torch.unsqueeze(target_boxes_weights, dim=-1)  # [N, M, C]

        encoded_target_boxes = self._box_coder.encode(target_boxes, anchors.boxes)  # [N, M, 4]
        decoded_box_preds = self._box_coder.decode(box_preds, anchors.boxes)  # [N, M, 4]

        for name, weight, fn in self._regression_losses:
            targets = encoded_target_boxes if fn.encoded else target_boxes
            preds = decoded_box_preds if fn.decoded else box_preds

            loss = fn(preds, targets)  # [N, M]
            loss *= target_boxes_weights  # [N, M]
            loss = torch.sum(loss, dim=-1)  # [N]
            loss = divide_no_nan(loss, num_matches)  # [N]
            reg_loss = loss * weight

        for name, weight, fn in self._classes_losses:
            loss = fn(class_preds, target_labels)  # [N, M, C]
            loss = torch.sum(loss, dim=-1)  # [N, M]
            loss *= target_labels_weights
            loss = torch.sum(loss, dim=-1)  # [N]
            loss = divide_no_nan(loss, num_matches)  # [N]
            class_loss = loss * weight

        for name, weight, fn in self._box_losses:
            targets = target_boxes if fn.decoded else encoded_target_boxes
            preds = decoded_box_preds if fn.decoded else encoded_target_boxes

            loss = fn(preds, targets)  # [N, M]
            loss *= target_boxes_weights  # [N, M]
            loss = torch.sum(loss, dim=-1)  # [N]
            loss = divide_no_nan(loss, num_matches)  # [N]
            box_loss = loss * weight

        return reg_loss, class_loss, box_loss

    def predict(self, inputs):
        images = inputs[0]  # [N, 3, H, W]

        # box_preds: [N, Ax4, H, W], class_preds: [N, AxC, H, W]
        box_preds, class_preds, strides = self._model(images)

        feature_map_sizes = [
            box_pred.shape[-2:] for box_pred in box_preds
        ]

        anchors = self._anchor_generator(feature_map_sizes, strides)

        box_preds = cat(box_preds, (-1, 4), 1)  # [N, M, 4]
        class_preds = cat(class_preds, (-1, self._num_classes), 1)  # [N, M, C]

        class_preds = torch.sigmoid(class_preds)

        class_preds = torch.reshape(class_preds, (class_preds.shape[0], -1))  # [N, MxC]
        _, top_indices = torch.topk(class_preds, self._max_detections, sorted=False)  # [N, MD]

        indices = top_indices // self._num_classes  # [N, MD]
        classes = top_indices % self._num_classes  # [N, MD]

        class_scores = torch.gather(class_preds, 1, top_indices)  # [N, MD]

        box_preds = torch.gather(box_preds, 1, indices.unsqueeze(-1).repeat(1, 1, 4))  # [N, MD, 4]
        anchor_boxes = anchors.boxes[indices]  # [N, MD, 4]

        box_preds = self._box_coder.decode(box_preds, anchor_boxes)  # [N, MD, 4]

        box_preds = torch.reshape(box_preds, (self._max_detections, 4))  # [MD, 4]
        classes = torch.reshape(classes, (self._max_detections,))  # [MD]
        class_scores = torch.reshape(class_scores, (self._max_detections,))  # [MD]

        selected_indices = torchvision.ops.nms(box_preds, class_scores, self._iou_threshold)

        box_preds = box_preds[selected_indices]
        class_preds = classes[selected_indices]

        selected_scores = class_scores[selected_indices]

        return box_preds, selected_scores, class_preds


def cat(inputs_list, shape, dim):
    return torch.cat(
        [inputs.permute(0, 2, 3, 1).reshape(inputs.shape[0], *shape) for inputs in inputs_list],
        dim=dim,
    )


def one_hot(inputs, num_classes):
    mask = inputs >= 0
    return F.one_hot(inputs.to(torch.int64).clamp(min=0), num_classes) * mask.unsqueeze(dim=-1)


def divide_no_nan(x, y):
    zero = torch.zeros((), dtype=y.dtype)
    one = torch.ones((), dtype=y.dtype)

    mask = y == zero

    y = torch.where(mask, one, y)
    return torch.where(mask, zero, x / y)
