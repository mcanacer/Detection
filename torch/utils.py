import torch
import torch.nn as nn
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def detection_visualizer(img, idx_to_class, bbox=None, pred=None, points=None):

    # Convert image to HWC if it is passed as a Tensor (0-1, CHW).
    if isinstance(img, torch.Tensor):
        img = (img * 255).permute(1, 2, 0)

    img_copy = np.array(img).astype("uint8")
    _, ax = plt.subplots(frameon=False)

    ax.axis("off")
    ax.imshow(img_copy)

    # fmt: off
    if points is not None:
        points_x = [t[0] for t in points]
        points_y = [t[1] for t in points]
        ax.scatter(points_x, points_y, color="yellow", s=24)

    if bbox is not None:
        for single_bbox in bbox:
            x0, y0, x1, y1 = single_bbox[:4]
            width = x1 - x0
            height = y1 - y0

            ax.add_patch(
                mpl.patches.Rectangle(
                    (x0, y0), width, height, fill=False, edgecolor=(1.0, 0, 0),
                    linewidth=4, linestyle="solid",
                )
            )
            if len(single_bbox) > 4:  # if class info provided
                obj_cls = idx_to_class[single_bbox[4].item()]
                ax.text(
                    x0, y0, obj_cls, size=18, family="sans-serif",
                    bbox={
                        "facecolor": "black", "alpha": 0.8,
                        "pad": 0.7, "edgecolor": "none"
                    },
                    verticalalignment="top",
                    color=(1, 1, 1),
                    zorder=10,
                )

    if pred is not None:
        for single_bbox in pred:
            x0, y0, x1, y1 = single_bbox[:4]
            width = x1 - x0
            height = y1 - y0

            ax.add_patch(
                mpl.patches.Rectangle(
                    (x0, y0), width, height, fill=False, edgecolor=(0, 1.0, 0),
                    linewidth=4, linestyle="solid",
                )
            )
            if len(single_bbox) > 4:  # if class info provided
                obj_cls = idx_to_class[single_bbox[4].item()]
                conf_score = single_bbox[5].item()
                ax.text(
                    x0, y0 + 15, f"{obj_cls}, {conf_score:.2f}",
                    size=18, family="sans-serif",
                    bbox={
                        "facecolor": "black", "alpha": 0.8,
                        "pad": 0.7, "edgecolor": "none"
                    },
                    verticalalignment="top",
                    color=(1, 1, 1),
                    zorder=10,
                )
    # fmt: on
    plt.show()


class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))
        else:
            self.mean = mean
        if std is None:
            self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        else:
            self.std = std
        if torch.cuda.is_available():
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()

    def forward(self, boxes, deltas):

        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes


class ClipBoxes(nn.Module):

    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes


class Anchors(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super().__init__()
        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        if strides is None:
            self.strides = [2 ** s for s in self.pyramid_levels]
        if sizes is None:
            self.sizes = [2 ** (s + 2) for s in self.pyramid_levels]
        if ratios is None:
            self.ratios = [0.5, 1, 2]
        if scales is None:
            self.scales = [2 ** (i/3) for i in range(3)]

    def forward(self, images):
        image_shape = images.shape[-2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** i - 1) // 2 ** i for i in self.pyramid_levels]

        all_anchors = np.zeros((0, 4)).astype(np.float32)

        for idx, p in enumerate(self.pyramid_levels):
            anchors = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

        all_anchors = np.expand_dims(all_anchors, axis=0)

        anchors = torch.from_numpy(all_anchors.astype(np.float32))
        if torch.cuda.is_available():
            anchors = anchors.cuda()
        return anchors


def generate_anchors(base_size, ratios, scales):
    num_anchors = len(ratios) * len(scales)

    anchors = np.zeros((num_anchors, 4))

    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    areas = anchors[:, 2] * anchors[:, 3]

    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[0]) + 0.5) * stride
    shift_y = (np.arange(0, shape[1]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K*A, 4))

    return all_anchors
