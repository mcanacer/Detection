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
            use_separable_conv=False,
            use_keras_layer=False,
            use_sync_bn=False,
            kernel_initializer='VarianceScaling',
            kernel_regularizer=None,
            bias_regularizer=None,
            **kwargs,
    ):
        self._config_dict = {
            'input_specs': input_specs,
            'min_level': min_level,
            'max_level': max_level,
            'num_filters': num_filters,
            'fusion_type': fusion_type,
            'use_separable_conv': use_separable_conv,
            'use_keras_layer': use_keras_layer,
            'activation': activation,
            'use_sync_bn': use_sync_bn,
            'norm_momentum': norm_momentum,
            'norm_epsilon': norm_epsilon,
            'kernel_initializer': kernel_initializer,
            'kernel_regularizer': kernel_regularizer,
            'bias_regularizer': bias_regularizer,
        }
        if use_separable_conv:
            conv2d = tf.keras.layers.SeparableConv2D
        else:
            conv2d = tf.keras.layers.Conv2D
        if use_sync_bn:
            norm = tf.keras.layers.experimental.SyncBatchNormalization
        else:
            norm = tf.keras.layers.BatchNormalization
        activation_fn = tf.keras.layers.Activation(activation)

        if tf.keras.backend.image_data_format() == 'channels_last':
            bn_axis = -1
        else:
            bn_axis = 1

        inputs = self._build_input_pyramid(input_specs, min_level)
        backbone_max_level = min(int(max(inputs.keys())), max_level)

        feats_lateral = {}
        for level in range(min_level, backbone_max_level + 1):
            feats_lateral[str(level)] = conv2d(
                filters=num_filters,
                kernel_size=1,
                padding='same',
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer)(
                inputs[str(level)])

        feats = {str(backbone_max_level): feats_lateral[str(backbone_max_level)]}
        for level in range(backbone_max_level - 1, min_level - 1, -1):
            feat_a = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(feats[str(level + 1)])
            feat_b = feats_lateral[str(level)]

            if fusion_type == 'sum':
                if use_keras_layer:
                    feats[str(level)] = tf.keras.layers.Add()([feat_a, feat_b])
                else:
                    feats[str(level)] = feat_a + feat_b
            elif fusion_type == 'concat':
                if use_keras_layer:
                    feats[str(level)] = tf.keras.layers.Concatenate(axis=-1)(
                        [feat_a, feat_b])
                else:
                    feats[str(level)] = tf.concat([feat_a, feat_b], axis=-1)
            else:
                raise ValueError('Fusion type {} not supported.'.format(fusion_type))

        for level in range(min_level, backbone_max_level + 1):
            feats[str(level)] = conv2d(
                filters=num_filters,
                strides=1,
                kernel_size=3,
                padding='same',
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer)(
                feats[str(level)])

        for level in range(backbone_max_level + 1, max_level + 1):
            feats_in = feats[str(level - 1)]
            if level > backbone_max_level + 1:
                feats_in = activation_fn(feats_in)
            feats[str(level)] = conv2d(
                filters=num_filters,
                strides=2,
                kernel_size=3,
                padding='same',
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer)(
                feats_in)

        for level in range(min_level, max_level + 1):
            feats[str(level)] = norm(
                axis=bn_axis, momentum=norm_momentum, epsilon=norm_epsilon)(
                feats[str(level)])

        self._output_specs = {
            str(level): feats[str(level)].shape
            for level in range(min_level, max_level + 1)
        }

        super(FPN, self).__init__(inputs=inputs, outputs=feats, **kwargs)

    def _build_input_pyramid(self, input_specs, min_level):
        assert isinstance(input_specs, dict)
        if min(input_specs.keys()) > str(min_level):
            raise ValueError(
                'Backbone min level should be less or equal to FPN min level')

        inputs = {}
        for level, spec in input_specs.items():
            inputs[level] = tf.keras.Input(shape=spec[1:])
        return inputs

    def get_config(self):
        return self._config_dict

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)

    @property
    def output_specs(self):
        """A dict of {level: TensorShape} pairs for the model output."""
        return self._output_specs