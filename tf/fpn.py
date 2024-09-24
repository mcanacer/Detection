import tensorflow as tf


class FPN(tf.keras.Model):

    def __init__(
            self,
            input_specs,
            min_level,
            max_level,
            num_filters=256,
            fusion_type='sum',
            activation='relu',
            norm_momentum=0.99,
            norm_epsilon=0.001,
            **kwargs,
    ):
        self._config_dict = {
            'input_specs': input_specs,
            'min_level': min_level,
            'max_level': max_level,
            'num_filters': num_filters,
            'fusion_type': fusion_type,
            'norm_momentum': norm_momentum,
            'norm_epsilon': norm_epsilon,
        }

        conv2d = tf.keras.layers.SeparableConv2D

        norm = tf.keras.layers.BatchNormalization

        if tf.keras.backend.image_data_format() == 'channel_last':
            self._bn_axis = -1
        else:
            self._bn_axis = 1

        self._activation = tf.keras.layers.Activation(activation)

        inputs = self._build_input_pyramid(input_specs)
        backbone_max_level = min(int(max(inputs.keys())), min_level)

        feats_lateral = {}
        for level in range(min_level, backbone_max_level + 1):
            feats_lateral[str(level)] = conv2d(
                filters=num_filters,
                kernel_size=1,
                padding='same',
            )(inputs[str(level)])

        feats = {str(backbone_max_level): feats_lateral[str(backbone_max_level)]}
        for level in range(backbone_max_level - 1, min_level - 1, -1):
            feat_a = tf.keras.layers.Upsample(size=(2, 2), interpolation='nearest')(feats[str(level + 1)])
            feat_b = feats_lateral[str(level)]

            if fusion_type == 'sum':
                feats[str(level)] = feat_a + feat_b
            elif fusion_type == 'concat':
                feats[str(level)] = tf.concat([feat_a, feat_b], axis=-1)
            else:
                raise ValueError("Fusion type {} not supported".format(fusion_type))

        for level in range(min_level, backbone_max_level + 1):
            feats[str(level)] = conv2d(
                filters=num_filters,
                strides=1,
                kernel_size=3,
                padding='same')(
                    feats[str(level)])

        for level in range(backbone_max_level + 1, max_level + 1):
            feats_in = feats[str(level - 1)]
            if level > backbone_max_level + 1:
                feats_in = self._activation(feats_in)
            feats[str(level)] = conv2d(
                filters=num_filters,
                strides=2,
                kernel_size=3,
                padding='same')(
                feats_in)

        for level in range(min_level, max_level + 1):
            feats[str(level)] = norm(
                axis=self._bn_axis, momentum=norm_momentum, epsilon=norm_epsilon)(
                feats[str(level)])

        self._output_specs = {
            str(level): feats[str(level)].shape
            for level in range(min_level, max_level + 1)
        }
        
        super(FPN, self).__init__(inputs=inputs, outputs=feats, **kwargs)

    def _build_input_pyramid(self, input_specs):
        inputs = {}
        for level, specs in input_specs.items():
            inputs[level] = tf.keras.Input(shape=specs[1:])
        return inputs
