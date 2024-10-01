import torch
from boxops import center


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

