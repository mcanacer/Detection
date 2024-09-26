import tensorflow as tf


class Assigner(object):

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
        for anchors_per_image, gt_boxes_per_image in zip(anchors.boxes, gt_boxes):
            similarities = self._similarity_calculator(gt_boxes_per_image, anchors_per_image)

            matched_values = tf.math.reduce_max(similarities, axis=0)
            matched_indices = tf.math.argmax(similarities, axis=0)

            neg_mask = matched_values < self._low_threshold
            ignore_mask = (matched_values < self._high_threshold) & (matched_values >= self._low_threshold)

            matched_indices[neg_mask] = -1
            matched_values[ignore_mask] = -2

            matched_idx.append(matched_indices)

        return matched_idx

