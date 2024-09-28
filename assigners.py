import torch


class ArgmaxAssigner(object):

    def __init__(
            self,
            similarity_calculator,
            high_threshold,
            low_threshold,
    ):
        self._similarity_calculator = similarity_calculator
        self._high_threshold = high_threshold
        self._low_threshold = low_threshold

    def __call__(self, gt_boxes, anchors):
        matched_idx = []
        anchors_per_image = anchors[0]
        for gt_boxes_per_image in gt_boxes:
            similarities = self._similarity_calculator(gt_boxes_per_image, anchors_per_image)

            matched_values, matched_indices = torch.max(similarities, dim=0)

            neg_mask = matched_values < self._low_threshold
            ignore_mask = (matched_values < self._high_threshold) & (matched_values >= self._low_threshold)

            matched_indices[neg_mask] = -1
            matched_values[ignore_mask] = -2

            matched_idx.append(matched_indices)

        return torch.stack(matched_idx, dim=0)

