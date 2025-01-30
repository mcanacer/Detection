import torch
import torch.nn.functional as F

import boxops


class FCOSAssigner(object):

    def __init__(self, ranges):
        self._ranges = ranges

    def __call__(
        self,
        points,
        gt_boxes,
        gt_labels,
        gt_weights,
    ):
        '''
        Args:
            points: dict per level [M, 2]
            gt_boxes: [N, T, 4]
            gt_labels: [N, T]
            gt_weights: [N, T]
        '''
        target_boxes = []
        target_boxes_weights = []
        target_labels = []

        for level, points_per_level in points.items():
            min_range, max_range = self._ranges[level]

            x_centers, y_centers = points_per_level.unbind(dim=1)

            x_centers = x_centers.unsqueeze(dim=0).unsqueeze(dim=-1)  # [1, M, 1]
            y_centers = y_centers.unsqueeze(dim=0).unsqueeze(dim=-1)  # [1, M, 1]

            # for each shape [N, 1, T] 
            x0, y0, x1, y1 = gt_boxes.unsqueeze(1).unbind(dim=-1)

            # distances: [N, M, T, 4]
            distances = torch.stack(
                [
                    x_centers - x0,
                    y_centers - y0,
                    x1 - x_centers,
                    y1 - y_centers,
                ],
                dim=-1,
            )

            # [N, M, T]
            matched_matrix = distances.min(dim=-1).values > 0  # inside gt_boxes

            # [N, M, T]
            matched_matrix &= (distances.max(dim=-1).values > min_range) & (distances.max(dim=-1).values < max_range)

            matched_matrix = matched_matrix.to(torch.float32)

            # [N, T]
            gt_areas = (gt_boxes[..., 2] - gt_boxes[..., 0]) * (gt_boxes[..., 3] - gt_boxes[..., 1])

            # [N, M, T]
            matched_matrix *= 1e8 - gt_areas.unsqueeze(dim=1)

            # matched_values: [N, M], matched_indices: [N, M]
            matched_values, matched_indices = matched_matrix.max(dim=-1)

            # [N, M]
            target_boxes_weights_per_level = matched_values > 0

            # [N. M]
            target_labels_per_level = torch.gather(gt_labels, 1, matched_indices)
            target_labels_per_level *= target_boxes_weights_per_level

            # [N, M, 4]
            target_boxes_per_level = torch.gather(gt_boxes, 1, matched_indices.unsqueeze(dim=-1).repeat(1, 1, 4))
            target_boxes_per_level *= target_boxes_weights_per_level.unsqueeze(dim=-1)

            target_labels.append(target_labels_per_level)
            target_boxes.append(target_boxes_per_level)
            target_boxes_weights.append(target_boxes_weights_per_level)

        target_labels = torch.cat(target_labels, dim=1)  # [N, sum of points]
        target_labels_weights = torch.ones_like(target_labels)  # [N, sum of points]
        target_boxes = torch.cat(target_boxes, dim=1)  # [N, points, 4]
        target_boxes_weights = torch.cat(target_boxes_weights, dim=1)  # [N, sum of points]

        points = torch.cat([points for points in points.values()], dim=0)

        num_matches = target_boxes_weights.sum(dim=1)  # [N]

        return target_boxes, target_boxes_weights, target_labels, target_labels_weights, points, num_matches



class ATSS(object):

    def __init__(self, k=9, s=8):
        self._k = k
        self._s = s


    def __call__(self, points, strides, num_points, gt_boxes, gt_labels, gt_weights):
        scales = torch.unsqueeze(strides, dim=-1) / (self._s / 2)  # [M, 1]
        anchors = torch.cat([points - scales, points + scales], dim=-1)  # [M, 4]

        distances = -boxops.center_distance(gt_boxes, anchors)  # [N, T, M]
        similarities = boxops.iou(gt_boxes, anchors)  # [N, T, M]

        mask = torch.full(similarities.shape, False)

        start = 0
        total = 0
        total_num_points = sum(num_points)
        for num_point in num_points:
            end = start + num_point

            k = min(self._k, end - start)

            level_distance = distances[..., start:end]

            _, indices = torch.topk(level_distance, k, dim=-1, sorted=False)  # [N, T, K]

            candidates = F.one_hot(start + indices, total_num_points)  # [N, T, K, M]

            candidates = torch.sum(candidates, dim=2)  # [N, T, M]

            mask |= candidates == 1

            total += k

            start = end

        similarities_mean = (torch.sum(similarities * mask, dim=-1, keepdims=True) / total)  # [N, T, 1]

        similarities_var = (torch.sum(
            torch.square((similarities - similarities_mean) * mask),
            dim=-1,
            keepdims=True,
            ) / total  # [N, T, 1]
        )

        similarities_std = torch.sqrt(similarities_var)  # [N, T, 1]

        iou_threshold = similarities_mean + similarities_std  # [N, T, 1]

        insides = boxops.inside(gt_boxes, anchors)  # [N, T]

        mask &= insides

        mask &= torch.unsqueeze(gt_weights == 1.0, dim=-1)  # [N, T, M]

        mask &= iou_threshold >= similarities

        similarities = torch.where(
            mask,
            similarities,
            torch.full(similarities.shape, -1.0)
        )  # [N, T, M]

        matched_values, matched_indices = torch.max(similarities, dim=1)  # [N, M]

        matched_mask = matched_values != -1.0  # [N, M]

        target_boxes = torch.gather(gt_boxes, 1, matched_indices.unsqueeze(dim=-1).repeat(1, 1, 4))  # [N, M, 4]
        target_boxes_weights = matched_mask  # [N, M]

        target_labels = torch.gather(gt_labels, 1, matched_indices)  # [N, M]
        target_labels *= matched_mask

        target_labels_weights = torch.ones_like(target_labels)  # [N, M]

        num_matches = torch.sum(matched_mask, dim=-1)  # [N]

        return target_boxes, target_boxes_weights, target_labels, target_labels_weights, num_matches

