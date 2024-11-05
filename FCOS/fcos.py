import torch
import torch.nn as nn
import torch.nn.functional as F

import losses
from boxops import scale, normalize

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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self._class_head._out_conv.weight.data.fill_(0)
        self._class_head._out_conv.bias.data.fill_(-math.log((1.0 - prior) / prior))

        # self._box_head._out_conv.weight.data.fill_(0)
        # self._box_head._out_conv.bias.data.fill_(0)

    def forward(self, inputs, training=None):
        inputs = self._backbone(inputs)
        inputs = self._fpn(inputs)

        inputs = {
            str(idx): inputs[idx] for idx in self._feature_map_indexes
        }

        class_preds = inputs

        class_preds = {
            idx: self._class_tower(class_pred, training=training)
            for idx, class_pred in class_preds.items()
        }

        class_preds = {
            idx: self._class_head(class_pred, training=training)
            for idx, class_pred in class_preds.items()
        }

        box_preds = inputs

        box_preds = {
            idx: self._box_tower(box_pred, training=training)
            for idx, box_pred in box_preds.items()
        }


        centerness_preds = {
            idx: self._centerness_head(box_pred)
            for idx, box_pred in box_preds.items()
        }

        box_preds = {
            idx: self._box_head(box_pred, training=training)
            for idx, box_pred in box_preds.items()
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
            location_generator,
            assigner,
            box_coder,
            max_detections=40,
            class_loss=[('focal', 1.0, losses.Focal())],
            reg_loss=[('L1', 1.0, losses.L1())],
            centerness_loss=[('centerness', 1.0, losses.CenternessLoss())],
    ):
        super(FCOS, self).__init__()

        self._model = model

        self._num_classes = num_classes
        self._location_generator = location_generator
        self._assigner = assigner
        self._box_coder = box_coder

        self._max_detections = max_detections

        self._class_loss = class_loss
        self._reg_loss = reg_loss
        self._centerness_loss = centerness_loss

    def forward(self, inputs, mode='losses'):
        return getattr(self, mode)(inputs)

    def losses(self, inputs):
        images = inputs[0]  # [N, 3, H, W]
        gt_boxes = inputs[1]  # [N, T, 4]
        gt_labels = inputs[2]  # [N, T]
        gt_weights = inputs[3]  # [N, T]

        # class_preds: [N, C, H, W], box_preds: [N. 4, H, W], box_quality: [N, 1, H, W]
        class_preds, box_preds, centerness_preds, strides = self._model(images, training=True)

        feature_map_sizes = [
            box_pred.shape[-2:] for box_pred in box_preds
        ]

        class_preds = cat(class_preds, (-1, self._num_classes), 1)  # [N, M, C]
        box_preds = cat(box_preds, (-1, 4), 1)  # [N, M, 4]
        centerness_preds = cat(centerness_preds, (-1, 1), 1)  # [N, M, 1]

        locations = self._location_generator(feature_map_sizes, strides)

        image_height, image_width = images.shape[-2:]

        gt_boxes = scale(gt_boxes, image_height, image_width)

        (
            target_boxes,  # [N, M, 4]
            target_labels,  # [N, M]
            target_centerness,  # [N, M, 1]
            target_weights,  # [N, M]
            locations,  # [M, 2]
            num_matches  # [N]
        ) = self._assigner(
            locations,  # [M, 2]
            gt_boxes,  # [N, T, 4]
            gt_labels,  # [N, T]
            gt_weights,  # [N, T]
        )

        target_labels = one_hot(target_labels - 1, self._num_classes)  # [N, M, C]
        target_labels *= target_weights.unsqueeze(dim=-1)

        encoded_target_boxes = self._box_coder.encode(target_boxes, locations)
        decoded_box_preds = self._box_coder.decode(box_preds, locations)

        encoded_target_boxes = normalize(encoded_target_boxes, image_height, image_width)

        print(num_matches)

        for name, weight, fn in self._reg_loss:
            targets = encoded_target_boxes if fn.encoded else target_boxes
            preds = decoded_box_preds if fn.decoded else box_preds

            losses = fn(preds, targets)  # [N, M]
            losses *= target_weights  # [N, M]
            losses = torch.sum(losses, dim=-1)  # [N]
            losses = divide_no_nan(losses, num_matches)  # [N] 
            box_loss = losses * weight  # [N]

        for name, weight, fn in self._class_loss:
            losses = fn(class_preds, target_labels)  # [N, M, C]
            losses = torch.sum(losses, dim=-1)  # [N, M]
            # losses *= target_weights  # [N, M]
            losses = torch.sum(losses, dim=-1)  # [N]
            losses = divide_no_nan(losses, num_matches)  # [N]
            class_loss = losses * weight  # [N]

        for name, weight, fn in self._centerness_loss:
            losses = fn(centerness_preds, target_centerness)  # [N, M]
            losses *= target_weights  # [N, M]
            losses = torch.sum(losses, dim=-1)  # [N]
            losses = divide_no_nan(losses, num_matches)  # [N]
            centerness_loss = losses * weight  # [N]

        return class_loss, box_loss, centerness_loss


def cat(inputs_list, shape, dim):
    return torch.cat(
        [inputs.permute(0, 2, 3, 1).contiguous().view(inputs.shape[0], *shape)
         for inputs in inputs_list],
         dim=dim,
    )


def one_hot(inputs, num_classes):
    mask = inputs >= 0
    return F.one_hot(inputs.clamp(min=0), num_classes) * mask.unsqueeze(dim=-1)


def divide_no_nan(x, y):
    zero = torch.zeros((), dtype=y.dtype)
    one = torch.ones((), dtype=y.dtype)

    mask = y == zero

    y = torch.where(mask, one, y)
    return torch.where(mask, zero, x / y)



