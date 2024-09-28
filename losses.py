import torch

import torch.nn.functional as F
from boxcoder import BoxCoder


def calc_iou(a, b):

    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    IoU = intersection / ua

    return IoU


def focal_loss(inputs, targets, alpha=0.25, gamma=2, reduction='none'):
    p = torch.sigmoid(inputs)
    bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = bce * (1 - p_t) ** gamma

    if alpha > 0:
        alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
        loss *= alpha_factor

    if reduction == 'none':
        pass
    elif reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


def l1_loss(box_coder, anchors_per_image, matched_gt_boxes_per_image, bbox_reg_per_image):
    target_reg = box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
    return F.l1_loss(bbox_reg_per_image, target_reg, reduction='sum')


class Focal(object):

    def __call__(self, cls_pred, gt_cls, matched_idx):
        losses = []
        for cls_pred_per_image, gt_cls_per_image, matched_idx_per_image in zip(cls_pred, gt_cls, matched_idx):
            pos_mask = matched_idx_per_image >= 0
            num_pos = pos_mask.sum()

            gt_labels = torch.zeros_like(cls_pred_per_image)
            gt_labels[pos_mask, gt_cls_per_image[matched_idx_per_image[pos_mask]].long()] = 1.0

            valid_idx_per_image = matched_idx_per_image != -2

            loss = focal_loss(
                inputs=cls_pred_per_image[valid_idx_per_image],
                targets=gt_labels[valid_idx_per_image],
                reduction='sum'
            )

            losses.append(loss / max(1, num_pos))

        return sum(losses)


class BoxLoss(object):

    def __init__(self):
        self._box_coder = BoxCoder()

    def __call__(self, box_pred, gt_boxes, anchors, matched_idx):
        losses = []
        for box_pred_per_image, gt_boxes_per_image, matched_idx_per_image in zip(box_pred, gt_boxes, matched_idx):
            anchors_per_image = anchors[0]

            pos_mask = matched_idx_per_image >= 0
            num_pos = pos_mask.sum()

            matched_gt_boxes_per_image = gt_boxes_per_image[matched_idx_per_image[pos_mask]]
            bbox_reg_per_image = box_pred_per_image[pos_mask, :]
            anchors_per_image = anchors_per_image[pos_mask, :]

            loss = l1_loss(
                self._box_coder,
                anchors_per_image,
                matched_gt_boxes_per_image,
                bbox_reg_per_image
            )

            losses.append(loss / max(1, num_pos))

        return sum(losses)

