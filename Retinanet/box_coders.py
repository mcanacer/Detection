import torch
from boxops import center, center_and_scale


class OffsetBoxCoder(object):

    def encode(self, boxes, anchors):
        cx, cy = center(anchors)
        xmin, ymin, xmax, ymax = torch.unbind(boxes, dim=-1)

        l = cx - xmin
        t = cy - ymin
        r = xmax - cx
        b = ymax - cy

        return torch.stack([l, t, r, b], dim=-1)

    def decode(self, encoded_boxes, anchors):
        cx, cy = center(anchors)
        l, t, r, b = torch.unbind(encoded_boxes, dim=-1)

        xmin = cx - l
        ymin = cy - t
        xmax = cx + r
        ymax = cy + b

        return torch.stack([xmin, ymin, xmax, ymax], dim=-1)


class FasterRcnnBoxCoder(object):

    def __init__(self, scales=None, eps=1e-8):
        self._scales = scales
        self._eps = eps

    def encode(self, boxes, anchors):
        xcenter_a, y_center_a, w_a, h_a = center_and_scale(anchors)
        xcenter, ycenter, w, h = center_and_scale(boxes)

        tx = (xcenter - xcenter_a) / w_a
        ty = (ycenter - y_center_a) / h_a
        tw = torch.log(w / w_a)
        th = torch.log(h / h_a)

        return torch.stack([tx, ty, tw, th], dim=-1)

    def decode(self, encoded_boxes, anchors):
        xcenter_a, ycenter_a, w_a, h_a = center_and_scale(anchors)
        tx, ty, tw, th = torch.unbind(encoded_boxes, dim=-1)

        w = torch.exp(tw) * w_a
        h = torch.exp(th) * h_a
        x_center = tx * w_a + xcenter_a
        y_center = ty * h_a + ycenter_a

        xmin = x_center - 0.5 * w
        ymin = y_center - 0.5 * h
        xmax = x_center + 0.5 * w
        ymax = y_center + 0.5 * h

        return torch.stack([xmin, ymin, xmax, ymax], dim=-1)

