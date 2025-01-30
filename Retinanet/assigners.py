import torch
import boxops
import torch.nn.functional as F


class ArgmaxAssigner(object):

    def __init__(
            self,
            similarity_calculator,
            matched_threshold,
            unmatched_threshold,
            num_classes,
    ):
        self._similarity_calculator = similarity_calculator
        self._matched_threshold = matched_threshold
        self._unmatched_threshold = unmatched_threshold
        self._num_classes = num_classes

    def __call__(self, anchors, gt_boxes, gt_labels, gt_weights):
        '''
        Args:
            anchors: [M, 4]
            gt_boxes: [N, T, 4]
            gt_labels: [N, T]
            gt_weights: [N, T]
        Return:
            target_boxes: [N, M, 4]
            target_boxes_weights: [N, M]
            target_labels: [N, M]
            target_labels_weights: [N, M]
            num_matches: [N]
        '''
        similarities = self._similarity_calculator(gt_boxes, anchors.boxes)  # [N, T, M]
        similarities *= torch.unsqueeze(gt_weights, dim=-1)

        # mathced_values: [N, M], matched_indices: [N, M]
        matched_values, matched_indices = torch.max(similarities, dim=1)

        matched_mask = matched_values >= self._matched_threshold  # [N, M]
        unmatched_mask = matched_values < self._unmatched_threshold  # [N, M]

        target_boxes = torch.gather(gt_boxes, 1, matched_indices.unsqueeze(2).repeat(1, 1, 4))  # [N, M, 4]
        target_boxes_weights = matched_mask  # [N, M]

        target_labels = torch.gather(gt_labels, 1, matched_indices)  # [N, M]
        target_labels *= matched_mask
        target_labels_weights = matched_mask | unmatched_mask  # [N, M]

        num_matches = matched_mask.sum(dim=1)  # [N]

        return target_boxes, target_boxes_weights, target_labels, target_labels_weights, num_matches


class ATSSAssigner(object):

    def __init__(self, k):
        self._k = k

    def __call__(self, anchors, gt_boxes, gt_labels, gt_weights):
        '''
        Args:
            anchors: [M, 4]
            gt_boxes: [N, T, 4]
            gt_labels: [N, T]
            gt_weights: [N, T]
        Return:
            target_boxes: [N, M, 4]
            target_boxes_weights: [N, M]
            target_labels: [N, M]
            target_labels_weights: [N, M]
            num_matches: [N]
        '''

        similarities = boxops.iou(gt_boxes, anchors.boxes)  # [N, T, M]
        distances = -boxops.center_distance(gt_boxes, anchors.boxes)  # [N, T, M]

        mask = torch.full(distances.shape, False)  # [N, T, M]

        total = 0
        for start, end in anchors.indices:
            level_distances = distances[..., start:end]
            _, indices = torch.topk(level_distances, self._k, dim=-1, sorted=False)  # [N, T, K]

            candidates = F.one_hot(start + indices, anchors.total)  # [N, T, K, M]
            candidates = torch.sum(candidates, dim=-2)  # [N, T, M]

            mask |= candidates == 1

            total += self._k

        similarities_mean = (torch.mean(similarities * mask, dim=-1, keepdims=True) / total)  # [N, T, 1]

        similarities_var = (torch.sum(
            torch.square((similarities - similarities_mean) * mask),
            dim=-1,
            keepdims=True,
        ) / total)  # [N, T, 1]

        similarities_std = torch.sqrt(similarities_var)  # [N, T, 1]

        iou_threshold = similarities_mean + similarities_std  # [N, T, 1]

        mask &= similarities >= iou_threshold

        insides = boxops.inside(gt_boxes, anchors.boxes)

        mask &= insides  # [N, T, M]

        mask &= torch.unsqueeze(gt_weights == 1.0, dim=-1)  # [N, T, M]

        similarities = torch.where(
            mask,
            similarities,
            torch.full(similarities.shape, -1.0)
        )  # [N, T, M]

        # matched_values: [N, M], matched_indices: [N, M]
        matched_values, matched_indices = torch.max(similarities, dim=-2)

        matched_mask = matched_values != -1.0  # [N, M]

        target_boxes = torch.gather(gt_boxes, 1, matched_indices.unsqueeze(dim=-1).repeat(1, 1, 4))  # [N, M, 4]
        target_boxes_weights = matched_mask  # [N, M]

        target_labels = torch.gather(gt_labels, 1, matched_indices)  # [N, M]
        target_labels *= matched_mask

        target_labels_weights = torch.ones_like(target_boxes_weights)  # [N, M]

        num_matches = torch.sum(matched_mask, dim=-1)  # [N]

        return target_boxes, target_boxes_weights, target_labels, target_labels_weights, num_matches


