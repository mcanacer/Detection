import torch


def center(boxes):
    xmin, ymin, xmax, ymax = torch.unbind(boxes, dim=-1)
    cx = (xmax + xmin) / 2
    cy = (ymax + ymin) / 2
    return cx, cy


def iou(boxes1, boxes2):
    boxes1_ndims = boxes1.ndim
    boxes2_ndims = boxes2.ndim

    zero = torch.zeros(())

    for _ in range(boxes2_ndims - 1):
        boxes1 = torch.unsqueeze(boxes1, dim=-2)

    for _ in range(boxes1_ndims - 1):
        boxes2 = torch.unsqueeze(boxes2, dim=0)

    xmin_b1, ymin_b1, xmax_b1, ymax_b1 = torch.unbind(boxes1, dim=-1)
    xmin_b2, ymin_b2, xmax_b2, ymax_b2 = torch.unbind(boxes2, dim=-1)

    width_b1 = torch.maximum(zero, xmax_b1 - xmin_b1)
    height_b1 = torch.maximum(zero, ymax_b1 - ymin_b1)

    width_b2 = torch.maximum(zero, xmax_b2 - xmin_b2)
    height_b2 = torch.maximum(zero, ymax_b2 - ymin_b2)

    area_b1 = width_b1 * height_b1
    area_b2 = width_b2 * height_b2

    xmin_intersection = torch.maximum(xmin_b1, xmin_b2)
    ymin_intersection = torch.maximum(ymin_b1, ymin_b2)
    xmax_intersection = torch.minimum(xmax_b1, xmax_b2)
    ymax_intersection = torch.minimum(ymax_b1, ymax_b2)

    width_inter = torch.maximum(zero, xmax_intersection - xmin_intersection)
    height_inter = torch.maximum(zero, ymax_intersection - ymin_intersection)

    area_inter = width_inter * height_inter

    union = area_b1 + area_b2 - area_inter

    iou = area_inter / union

    return iou
