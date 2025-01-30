import torch
import math


def area(boxes):
    # boxes: [N, T, 4]
    xmin, ymin, xmax, ymax = torch.unbind(boxes, dim=-1)
    areas = (xmax - xmin) * (ymax - ymin)
    return areas


def center(boxes):
    xmin, ymin, xmax, ymax = torch.unbind(boxes, dim=-1)
    cx = 0.5 * (xmax + xmin)
    cy = 0.5 * (ymax + ymin)
    return cx, cy

def center_and_scale(boxes):
    xmin, ymin, xmax, ymax = torch.unbind(boxes, dim=-1)
    cx = 0.5 * (xmax + xmin)
    cy = 0.5 * (ymax + ymin)
    w = (xmax - xmin)
    h = (ymax - ymin)
    return cx, cy, w, h

def divide_no_nan(x, y):
    zero = torch.zeros((), dtype=y.dtype)
    one = torch.ones((), dtype=y.dtype)

    mask = y == zero

    y = torch.where(mask, one, y)
    return torch.where(mask, zero, x / y)


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

    iou = divide_no_nan(area_inter, union)

    return torch.maximum(iou, zero)


def giou(boxes1, boxes2):
    zero = torch.zeros(())

    b1_xmin, b1_ymin, b_xmax, b1_ymax = torch.unbind(boxes1, dim=-1)
    b2_xmin, b2_ymin, b2_xmax, b2_ymax = torch.unbind(boxes2, dim=-1)

    b1_w = torch.maximum(zero, b1_xmax - b1_xmin)
    b1_h = torch.maximum(zero, b2_ymax - b2_ymin)

    b1_area = b1_w * b1_h

    b2_w = torch.maximum(zero, b2_xmax - b2_xmin)
    b2_h = torch.maximum(zero, b2_ymax - b2_ymin)

    b2_area = b2_w * b2_h

    inter_xmin = torch.maximum(b1_xmin, b2_xmin)
    inter_ymin = torch.maximum(b1_ymin, b2_ymin)
    inter_xmax = torch.minimum(b1_xmax, b2_xmax)
    inter_ymax = torch.minimum(b1_ymax, b2_ymax)

    inter_w = torch.maximum(zero, inter_xmax - inter_xmin)
    inter_h = torch.maximum(zero, inter_ymax - inter_ymin)

    inter_area = inter_w * inter_h

    union = b1_area + b2_area - inter_area

    iou = divide_no_nan(inter_area, union)

    enclose_xmin = torch.minimum(b1_xmin, b2_xmin)
    enclose_ymin = torch.minimum(b1_ymin, b2_ymin)
    enclose_xmax = torch.maximum(b1_xmax, b2_xmax)
    enclose_ymax = torch.maximum(b1_ymax, b2_ymax)

    enclose_w = torch.maximum(zero, enclose_xmax - enclose_xmin)
    enclose_h = torch.maximum(zero, enclose_ymax - enclose_ymin)

    enclose_area = enclose_w * enclose_h

    giou = iou - divide_no_nan(enclose_area - union, enclose_area)

    return giou


def diou(boxes1, boxes2):
    boxes1_ndims = boxes1.ndim
    boxes2_ndims = boxes2.ndim

    zero = torch.zeros(())

    b1_xmin, b1_ymin, b1_xmax, b1_ymax = torch.unbind(boxes1, dim=-1)
    b2_xmin, b2_ymin, b2_xmax, b2_ymax = torch.unbind(boxes2, dim=-1)

    b1_w = torch.maximum(zero, b1_xmax - b1_xmin)
    b1_h = torch.maximum(zero, b2_ymax - b2_ymin)

    b1_area = b1_w * b1_h

    b2_w = torch.maximum(zero, b2_xmax - b2_xmin)
    b2_h = torch.maximum(zero, b2_ymax - b2_ymin)

    b2_area = b2_w * b2_h

    inter_xmin = torch.maximum(b1_xmin, b2_xmin)
    inter_ymin = torch.maximum(b1_ymin, b2_ymin)
    inter_xmax = torch.minimum(b1_xmax, b2_xmax)
    inter_ymax = torch.minimum(b1_ymax, b2_ymax)

    inter_w = torch.maximum(zero, inter_xmax - inter_xmin)
    inter_h = torch.maximum(zero, inter_ymax - inter_ymin)

    inter_area = inter_w * inter_h

    union = b1_area + b2_area - inter_area

    iou = divide_no_nan(inter_area, union)

    enclose_xmin = torch.minimum(b1_xmin, b2_xmin)
    enclose_ymin = torch.minimum(b1_ymin, b2_ymin)
    enclose_xmax = torch.maximum(b1_xmax, b2_xmax)
    enclose_ymax = torch.maximum(b1_ymax, b2_ymax)

    enclose_w = torch.maximum(zero, enclose_xmax - enclose_xmin)
    enclose_h = torch.maximum(zero, enclose_ymax - enclose_ymin)

    diagonal_distance = enclose_w ** 2.0 + enclose_h ** 2.0

    center_distance = (((b2_xmax + b2_xmin - b1_xmax - b1_xmin) ** 2.0 + (b2_ymax + b2_ymin - b1_ymax - b1_ymin) ** 2.0) / 4.0)

    diou = iou - divide_no_nan(center_distance, diagonal_distance)

    return diou


def ciou(boxes1, boxes2):
    zero = torch.zeros(())

    b1_xmin, b1_ymin, b1_xmax, b1_ymax = torch.unbind(boxes1, dim=-1)
    b2_xmin, b2_ymin, b2_xmax, b2_ymax = torch.unbind(boxes2, dim=-1)

    b1_w = torch.maximum(zero, b1_xmax - b1_xmin)
    b1_h = torch.maximum(zero, b1_ymax - b1_ymin)

    b1_area = b1_w * b1_h

    b2_w = torch.maximum(zero, b2_xmax - b2_xmin)
    b2_h = torch.maximum(zero, b2_ymax - b2_ymin)

    b2_area = b2_w * b2_h

    inter_xmin = torch.maximum(b1_xmin, b2_xmin)
    inter_ymin = torch.maximum(b1_ymin, b2_ymin)
    inter_xmax = torch.minimum(b1_xmax, b2_xmax)
    inter_ymax = torch.minimum(b1_ymax, b2_ymax)

    inter_w = torch.maximum(zero, inter_xmax - inter_xmin)
    inter_h = torch.maximum(zero, inter_ymax - inter_ymin)

    inter_area = inter_w * inter_h

    union = b1_area + b2_area - inter_area

    iou = divide_no_nan(inter_area, union)

    enclose_xmin = torch.minimum(b1_xmin, b2_xmin)
    enclose_ymin = torch.minimum(b1_ymin, b2_ymin)
    enclose_xmax = torch.maximum(b1_xmax, b2_xmax)
    enclose_ymax = torch.maximum(b1_ymax, b2_ymax)

    enclose_w = torch.maximum(zero, enclose_xmax - enclose_xmin)
    enclose_h = torch.maximum(zero, enclose_ymax - enclose_ymin)

    diagonal_distance = enclose_w ** 2.0 + enclose_h ** 2.0

    center_distance = (((b2_xmax + b2_xmin - b1_xmax - b1_xmin) ** 2.0 + (b2_ymax + b2_ymin - b1_ymax - b1_ymin) ** 2.0) / 4.0)

    b1_ratio = divide_no_nan(b1_w, b1_h)
    b2_ratio = divide_no_nan(b2_w, b2_h)

    v = (((torch.atan(b2_ratio) - torch.atan(b1_ratio)) ** 2.0) * (4.0 / (math.pi**2.0)))

    alpha = divide_no_nan(v, 1. - iou + v)

    ciou = iou - divide_no_nan(center_distance, diagonal_distance) - alpha * v

    return ciou


def center_distance(boxes1, boxes2):
    boxes1_ndims = boxes1.ndim
    boxes2_ndims = boxes2.ndim

    for _ in range(boxes2_ndims - 1):
        boxes1 = torch.unsqueeze(boxes1, dim=-2)

    for _ in range(boxes1_ndims - 1):
        boxes2 = torch.unsqueeze(boxes2, dim=0)

    cx_b1, cy_b1 = center(boxes1)
    cx_b2, cy_b2 = center(boxes2)

    x_distance = torch.square(cx_b1 - cx_b2)
    y_distance = torch.square(cy_b1 - cy_b2)

    return torch.sqrt(x_distance + y_distance)


def inside(boxes1, boxes2):
    boxes1_ndims = boxes1.ndim
    boxes2_ndims = boxes2.ndim

    for _ in range(boxes2_ndims - 1):
        boxes1 = torch.unsqueeze(boxes1, dim=-2)

    for _ in range(boxes1_ndims - 1):
        boxes2 = torch.unsqueeze(boxes2, dim=0)

    xmin_b1, ymin_b1, xmax_b1, ymax_b1 = torch.unbind(boxes1, dim=-1)

    cx_b2, cy_b2 = center(boxes2)

    distances = torch.stack(
        [
            cx_b2 - xmin_b1,
            cy_b2 - ymin_b1,
            xmax_b1 - cx_b2,
            ymax_b1 - cy_b2
        ],
        dim=-1,
    )

    return torch.min(distances, dim=-1).values > 0.0





