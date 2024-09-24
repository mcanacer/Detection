import tensorflow as tf
import numpy as np


class RetinaNetHead(tf.keras.layers.Layer):

    def __init__(
            self,
            min_level,
            max_level,
            num_classes,
            num_anchors_per_location,
            num_convs=4,
            activation='relu',
            num_filters=256,
            norm_momentum=0.99,
            norm_epsilon=0.001,
            **kwargs,
    ):
        super(RetinaNetHead, self).__init__(**kwargs)
        self._config_dict = {
            'min_level': min_level,
            'max_level': max_level,
            'num_classes': num_classes,
            'num_anchors_per_location': num_anchors_per_location,
            'num_convs': num_convs,
            'norm_momentum': norm_momentum,
            'norm_epsilon': norm_epsilon,
            'num_filters': num_filters,
        }

        if tf.keras.backend.image_data_format() == 'channels_last':
            self._bn_axis = -1
        else:
            self._bn_axis = 1

        self._activation = tf.keras.layers.Activation(activation)

    def build(self, input_shape):
        conv_op = tf.keras.layers.SeparableConv2D

        conv_kwargs = {
            'filters': self._config_dict['num_filters'],
            'kernel_size': 3,
            'padding': 'same',
            'bias_initializer': tf.zeros_initializer(),
        }

        bn_op = tf.keras.layers.BatchNormalization

        bn_kwargs = {
            'axis': self._bn_axis,
            'momentum': self._config_dict['norm_momentum'],
            'epsilon': self._config_dict['norm_epsilon'],
        }

        self._cls_convs = []
        self._cls_norms = []

        for level in range(self._config_dict['min_level'], self._config_dict['max_level'] + 1):
            this_level_cls_norms = []
            for i in range(self._config_dict['num_convs']):
                if level == self._config_dict['min_level']:
                    cls_conv_name = 'classnet-conv_{}'.format(i)
                    self._cls_convs.append(conv_op(name=cls_conv_name, **conv_kwargs))
                cls_norm_name = 'classnet_conv-norm_{}_{}'.format(level, i)
                this_level_cls_norms.append(bn_op(name=cls_norm_name, **bn_kwargs))
            self._cls_norms.append(this_level_cls_norms)

        classifier_kwargs = {
            'filters': (
                    self._config_dict['num_classes'] *
                    self._config_dict['num_anchors_per_location']),
            'kernel_size': 3,
            'padding': 'same',
            'bias_initializer': tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
        }

        self._classifier = conv_op(name='scores', **classifier_kwargs)

        self._box_convs = []
        self._box_norms = []

        for level in range(self._config_dict['min_level'], self._config_dict['max_level'] + 1):
            this_level_box_norms = []
            for i in range(self._config_dict['num_convs']):
                if level == self._config_dict['min_level']:
                    box_conv_name = 'boxnet_conv_{}'.format(i)
                    self._box_convs.append(conv_op(name=box_conv_name, **conv_kwargs))
                box_norm_name = 'boxnet_conv_norm_{}_{}'.format(level, i)
                this_level_box_norms.append(bn_op(name=box_norm_name, **bn_kwargs))
            self._box_norms.append(this_level_box_norms)

        box_regressor_kwargs = {
            'filters': (self._config_dict['num_anchors_per_location'] * 4),
            'kernel_size': 3,
            'padding': 'same',
            'bias_initializer': tf.zeros_initializer(),
        }

        self._box_regressor = conv_op(name='boxes', **box_regressor_kwargs)
        super(RetinaNetHead, self).build(input_shape)

    def call(self, features, training=None):
        scores = {}
        boxes = {}

        for idx, level in enumerate(range(self._config_dict['min_level'], self._config_dict['max_level'] + 1)):
            this_level_features = features[str(level)]

            x = this_level_features
            for conv, norm in zip(self._cls_convs, self._cls_norms[idx]):
                x = conv(x)
                x = norm(x)
                x = self._activation(x)
            classnet_x = x
            scores[str(level)] = self._classifier(classnet_x)

            x = this_level_features
            for conv, norm in zip(self._box_convs, self._box_norms[idx]):
                x = conv(x)
                x = norm(x)
                x = self._activation(x)
            boxes[str(level)] = self._box_regressor(x)

        return scores, boxes

    def get_config(self):
        return self._config_dict
