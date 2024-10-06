import torch


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
        similarities = self._similarity_calculator(gt_boxes, anchors)  # [N, T, M]
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

        num_matches = matched_mask.sum(dim=1).clamp(min=1)  # [N]

        return target_boxes, target_boxes_weights, target_labels, target_labels_weights, num_matches





