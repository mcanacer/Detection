import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import losses
from boxops import scale, normalize, giou, ciou

import math


class FeatureExtractor(nn.Module):

    def __init__(
            self,
            backbone,
            fpn,
            class_tower,
            box_tower,
            class_head,
            box_head,
            centerness_head,
            feature_map_indexes,
    ):
        super(FeatureExtractor, self).__init__()

        self._backbone = backbone
        self._fpn = fpn

        self._class_tower = class_tower
        self._box_tower = box_tower
        self._class_head = class_head
        self._box_head = box_head
        self._centerness_head = centerness_head

        self._feature_map_indexes = feature_map_indexes

        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False
                layer.eval()

        prior = 0.01

        self._class_head._out_conv.bias.data.fill_(math.log(prior / (1.0 - prior)))


    def forward(self, inputs):
        inputs = self._backbone(inputs)
        inputs = self._fpn(inputs)

        inputs = {
            str(idx): inputs[idx] for idx in self._feature_map_indexes
        }

        box_preds = inputs

        box_preds = self._box_tower(box_preds)

        centerness_preds = {
            idx: self._centerness_head(box_pred)
            for idx, box_pred in box_preds.items()
        }

        box_preds = {
            idx: self._box_head(box_pred)
            for idx, box_pred in box_preds.items()
        }

        class_preds = inputs

        class_preds = self._class_tower(class_preds)

        class_preds = {
            idx: self._class_head(class_pred)
            for idx, class_pred in class_preds.items()
        }

        class_preds = [class_preds[idx] for idx in sorted(class_preds.keys())]
        box_preds = [box_preds[idx] for idx in sorted(box_preds.keys())]
        centerness_preds = [centerness_preds[idx] for idx in sorted(centerness_preds.keys())]

        strides = {idx: 2 ** idx for idx in sorted(self._feature_map_indexes)}

        return class_preds, box_preds, centerness_preds, strides



class FCOS(nn.Module):

    def __init__(
            self,
            model,
            num_classes,
            points_generator,
            assigner,
            box_coder,
            max_detections=40,
            iou_threshold=0.5,
            score_threshold=0.5,
            class_loss=[('Focal', 1.0, losses.Focal())],
            reg_loss=[('Reg', 0.1, losses.L1())],
            box_loss=[('Centerness', 1.0, losses.CenternessLoss())],
    ):
        super(FCOS, self).__init__()

        self._model = model

        self._num_classes = num_classes
        self._points_generator = points_generator
        self._assigner = assigner
        self._box_coder = box_coder

        self._max_detections = max_detections
        self._iou_threshold = iou_threshold
        self._score_threshold = score_threshold

        self._class_loss = class_loss
        self._reg_loss = reg_loss
        self._box_loss = box_loss

    def forward(self, inputs, mode='losses'):
        return getattr(self, mode)(inputs)

    def losses(self, inputs):
        images = inputs[0]  # [N, 3, H, W]
        gt_boxes = inputs[1]  # [N, T, 4]
        gt_labels = inputs[2]  # [N, T]
        gt_weights = inputs[3]  # [N, T]

        # class_preds: [N, C, H, W], box_preds: [N. 4, H, W], box_quality: [N, 1, H, W]
        class_preds, box_preds, centerness_preds, strides = self._model(images)

        feature_map_sizes = [
            box_pred.shape[-2:] for box_pred in box_preds
        ]

        class_preds = cat(class_preds, (-1, self._num_classes), 1)  # [N, M, C]
        box_preds = cat(box_preds, (-1, 4), 1)  # [N, M, 4]
        centerness_preds = cat(centerness_preds, (-1, 1), 1)  # [N, M, 1]

        points, strides = self._points_generator(feature_map_sizes, strides)  # [M, 2]

        strides = torch.reshape(strides, (1, -1, 1))  # [1, M, 1]

        image_height, image_width = images.shape[-2:]

        gt_boxes = scale(gt_boxes, image_height, image_width)

        (
            target_boxes,  # [N, M, 4]
            target_boxes_weights,  # [N, M]
            target_labels,  # [N, M]
            target_labels_weights,  # [N, M]
            points,  # [M, 2]
            num_matches  # [N]
        ) = self._assigner(
            points,  # [M, 2]
            gt_boxes,  # [N, T, 4]
            gt_labels,  # [N, T]
            gt_weights,  # [N, T]
        )

        target_labels = one_hot(target_labels - 1, self._num_classes)  # [N, M, C]
        target_labels *= target_boxes_weights.unsqueeze(dim=-1)

        encoded_target_boxes = self._box_coder.encode(target_boxes, points)
        encoded_target_boxes /= strides

        target_centerness = compute_centerness(encoded_target_boxes)

        decoded_box_preds = self._box_coder.decode(box_preds*strides, points)

        for name, weight, fn in self._reg_loss:
            targets = encoded_target_boxes if fn.encoded else target_boxes
            preds = decoded_box_preds if fn.decoded else box_preds

            loss = fn(preds, targets)  # [N, M]
            loss *= target_boxes_weights  # [N, M]
            loss = torch.sum(loss, dim=-1)  # [N]
            loss = divide_no_nan(loss, num_matches)  # [N] 
            box_loss = loss * weight  # [N]

        for name, weight, fn in self._class_loss:
            loss = fn(class_preds, target_labels)  # [N, M, C]
            loss = torch.sum(loss, dim=-1)  # [N, M]
            loss *= target_labels_weights
            loss = torch.sum(loss, dim=-1)  # [N]
            loss = divide_no_nan(loss, num_matches)  # [N]
            class_loss = loss * weight  # [N]

        for name, weight, fn in self._box_loss:
            loss = fn(centerness_preds, target_centerness)  # [N, M]
            loss *= target_boxes_weights  # [N, M]
            loss = torch.sum(loss, dim=-1)  # [N]
            loss = divide_no_nan(loss, num_matches)  # [N]
            centerness_loss = loss * weight  # [N]

        return box_loss, class_loss, centerness_loss

    def predict(self, inputs):
        images = inputs[0]  # [N, 3, H, W]

        # class_preds: [N, C, H, W], box_preds: [N. 4, H, W], box_quality: [N, 1, H, W]
        class_preds, box_preds, centerness_preds, strides = self._model(images)

        feature_map_sizes = [
            box_pred.shape[-2:] for box_pred in box_preds
        ]

        class_preds = cat(class_preds, (-1, self._num_classes), 1)  # [N, M, C]
        box_preds = cat(box_preds, (-1, 4), 1)  # [N, M, 4]
        centerness_preds = cat(centerness_preds, (-1, 1), 1)  # [N, M, 1]

        points, strides = self._points_generator(feature_map_sizes, strides)

        strides = torch.reshape(strides, (-1, 1))

        points = torch.cat([v for v in points.values()], dim=0) # [M, 2]

        image_height, image_width = images.shape[-2:]

        class_preds = torch.sqrt(class_preds.sigmoid_() * centerness_preds.sigmoid_())  # [N, M, C]

        class_preds = torch.reshape(class_preds, (class_preds.shape[0], -1))  # [N, MxC]

        _, top_indices = torch.topk(class_preds, self._max_detections, sorted=False)  # [N, MD]

        indices = top_indices // self._num_classes  # [N, MD]
        classes = top_indices % self._num_classes  # [N, MD]

        class_scores = torch.gather(class_preds, 1, indices)  # [N, MD]

        box_preds = torch.gather(box_preds, 1, indices.unsqueeze(dim=-1).repeat(1, 1, 4))  # [N, MD, 4]
        points = points[indices]  # [N, MD, 2]

        strides = strides[indices]

        box_preds = self._box_coder.decode(box_preds*strides, points)  # [N, MD, 4]

        box_preds = torch.reshape(box_preds, (self._max_detections, 4))  # [MD, 4]
        class_scores = torch.reshape(class_scores, (self._max_detections,))  # [MD]
        classes = torch.reshape(classes, (self._max_detections,))  # [MD]

        selected_indices = torchvision.ops.nms(box_preds, class_scores, self._iou_threshold)

        box_preds = box_preds[selected_indices]
        class_preds = classes[selected_indices]
        selected_scores = class_scores[selected_indices]

        return box_preds, selected_scores, class_preds


def cat(inputs_list, shape, dim):
    return torch.cat(
        [inputs.permute(0, 2, 3, 1).reshape(inputs.shape[0], *shape)
         for inputs in inputs_list],
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


def compute_centerness(encodings):
    lr = encodings[..., 0::2]
    tb = encodings[..., 1::2]

    min_lr = torch.min(lr, dim=-1)[0]
    max_lr = torch.max(lr, dim=-1)[0]

    min_tb = torch.min(tb, dim=-1)[0]
    max_tb = torch.max(tb, dim=-1)[0]

    centerness = (min_lr * min_tb) / (max_lr * max_tb)

    return torch.unsqueeze(torch.sqrt(centerness), dim=-1)

