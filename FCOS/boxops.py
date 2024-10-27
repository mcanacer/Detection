import torch


def area(boxes):
    # boxes: [N, T, 4]
    xmin, ymin, xmax, ymax = torch.unbind(boxes, dim=-1)
    areas = (xmax - xmin) * (ymax - ymin)
    return areas


def center(boxes):
    xmin, ymin, xmax, ymax = torch.unbind(boxes, dim=-1)
    cx = (xmax + xmin) / 2
    cy = (ymax + ymin) / 2
    return cx, cy


def scale(boxes, height, width):
    return boxes * torch.tensor([width, height, width, height], dtype=torch.float32)


def normalize(boxes, height, width):
    return boxes / torch.tensor([width, height, width, height], dtype=torch.float32)
