import torch


class FCOSAssigner(object):

    def __init__(self, box_coder, ranges):
        self._box_coder = box_coder
        self._ranges = ranges

    def __call__(
        self,
        locations,
        gt_boxes,
        gt_labels,
        gt_weights,
    ):
        '''
        Args:
            locations: dict per level [M, 2]
            gt_boxes: [N, T, 4]
            gt_labels: [N, T]
            gt_weights: [N, T]
        '''
        target_labels = []
        target_boxes = []
        target_weights = []

        for level, locations_per_level in locations.items():
            min_range, max_range = self._ranges[level]

            # M refers to number of location in feature level
            # x_centers: [M], y_centers: [M]
            x_centers, y_centers = locations_per_level.unbind(dim=1)

            x_centers = x_centers.view(1, -1, 1)  # [1, M, 1]
            y_centers = y_centers.view(1, -1, 1)  # [1, M, 1]

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
            target_weights_per_level = matched_values > 0

            # [N. M]
            target_labels_per_level = torch.gather(gt_labels, 1, matched_indices)
            target_labels_per_level *= target_weights_per_level

            # [N, M, 4]
            target_boxes_per_level = torch.gather(gt_boxes, 1, matched_indices.unsqueeze(dim=-1).repeat(1, 1, 4))
            target_boxes_per_level *= target_weights_per_level.unsqueeze(dim=-1)

            target_labels.append(target_labels_per_level)
            target_boxes.append(target_boxes_per_level)
            target_weights.append(target_weights_per_level)

        target_labels = torch.cat(target_labels, dim=1)  # [N, sum of locations]
        target_boxes = torch.cat(target_boxes, dim=1)  # [N, sum of locations, 4]
        target_weights = torch.cat(target_weights, dim=1)  # [N, sum of locations]

        locations = torch.cat([locations_per_level for level, locations_per_level in locations.items()], dim=0)

        ltrb = self._box_coder.encode(target_boxes, locations)  # [N, M, 4]

        target_centerness = self._calculate_centerness(ltrb)  # [N, M, 1]
        target_centerness *= target_weights.unsqueeze(dim=-1)

        num_matches = target_weights.sum(dim=1)  # [N]

        return target_boxes, target_labels, target_centerness, target_weights, locations, num_matches


    def _calculate_centerness(self, ltrb):
        lr = ltrb[:, :, 0::2]  # [N, M]
        tb = ltrb[:, :, 1::2]  # [N, M]

        c = ((lr.min(dim=2).values * tb.min(dim=2).values) / (lr.max(dim=2).values * tb.max(dim=2).values)).unsqueeze(dim=-1)

        return torch.sqrt(c)






