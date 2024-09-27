import torch
import torch.nn as nn

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

            gt_labels = torch.zeros_like(gt_cls_per_image)
            gt_labels[pos_mask, gt_cls_per_image[matched_idx_per_image[pos_mask]]] = 1.0

            valid_idx_per_image = matched_idx_per_image != -2

            loss = focal_loss(
                inputs=cls_pred_per_image[valid_idx_per_image],
                targets=gt_labels[valid_idx_per_image],
                reduction='sum'
            )

            losses.append(loss / max(1, num_pos))

        return losses


class BoxLoss(object):

    def __init__(self):
        self._box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

    def __call__(self, box_pred, gt_boxes, anchors, matched_idx):
        losses = []
        for box_pred_per_image, gt_boxes_per_image, anchors_per_image, matched_idx_per_image in zip(box_pred, gt_boxes, anchors, matched_idx):
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

        return losses


class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, classifications, reggressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]

        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

        for i in range(batch_size):

            classification = classifications[i, :, :]
            reggression = reggressions[i, :, :]

            bbox_annotation = annotations[i, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                    classification_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())
                    classification_losses.append(torch.tensor(0).float())
                continue

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)

            targets = torch.ones(classification.shape) * -1
            if torch.cuda.is_available():
                targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones(targets.shape) * alpha
            if torch.cuda.is_available():
                alpha_factor = alpha_factor.cuda()

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets*torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = bce * focal_weight

            zeros = torch.zeros(cls_loss.shape)
            if torch.cuda.is_available():
                zeros = zeros.cuda()

            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))

            if num_positive_anchors > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                gt_widths = torch.clamp(gt_widths, min=0.0)
                gt_heights = torch.clamp(gt_heights, min=0.0)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                norm = torch.Tensor([[0.1, 0.1, 0.2, 0.2]])
                if torch.cuda.is_available():
                    norm = norm.cuda()

                targets = targets / norm

                reggression_diff = torch.abs(targets - reggression[positive_indices])

                reg_loss = torch.where(torch.le(reggression_diff, 1.0 / 9.0), 0.5 * 9.0 * torch.pow(reggression_diff, 2), reggression_diff - 5.0 / 9.0)

                regression_losses.append(reg_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())

        return torch.stack(classification_losses).mean(dim=0, keepdims=True), torch.stack(regression_losses).mean(dim=0, keepdims=True)


