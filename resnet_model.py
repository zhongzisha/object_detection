# -*- coding: utf-8 -*-
# @Time    : 2019/11/15 9:36
# @Author  : Zisha Zhong
# @Email   : zhongzisha@outlook.com
# @File    : resnet_model.py
# @Software: PyCharm

"""RetinaNet (via ResNet) model definition.

Defines the RetinaNet model and loss functions from this paper:

https://arxiv.org/pdf/1708.02002

Uses the ResNet model as a basis.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

_WEIGHT_DECAY = 1e-4
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-4
_RESNET_MAX_LEVEL = 5


def drop_connect(inputs, is_training, survival_prob):
    if not is_training:
        return inputs

    batch_size = tf.shape(inputs)[0]
    random_tensor = survival_prob
    random_tensor += tf.random.uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.math.divide(inputs, survival_prob) * binary_tensor
    return output


def group_normalization(x, group=32, gamma_initializer=tf.constant_initializer(1.),
                        data_format='channels_first', name='gn'):
    """
    More code that reproduces the paper can be found at https://github.com/ppwwyyxx/GroupNorm-reproduce/.
    """
    with tf.variable_scope(name):
        shape = x.get_shape().as_list()
        if data_format == 'channels_first':
            chan = shape[1]
        else:
            chan = shape[3]
        group_size = chan // group

        orig_shape = tf.shape(x)
        if data_format == 'channels_first':
            h, w = orig_shape[2], orig_shape[3]
        else:
            h, w = orig_shape[1], orig_shape[2]

        if data_format == 'channels_first':
            x = tf.reshape(x, tf.stack([-1, group, group_size, h, w]))
            mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
            new_shape = [1, group, group_size, 1, 1]
        else:
            x = tf.reshape(x, tf.stack([-1, h, w, group, group_size]))
            mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
            new_shape = [1, 1, 1, group, group_size]

        beta = tf.get_variable('beta', [chan], initializer=tf.constant_initializer())
        gamma = tf.get_variable('gamma', [chan], initializer=gamma_initializer)

        beta = tf.reshape(beta, new_shape)
        gamma = tf.reshape(gamma, new_shape)
        out = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5, name='output')
        return tf.reshape(out, orig_shape, name='output')


def batch_norm_relu(inputs,
                    is_training_bn,
                    relu=True,
                    init_zero=False,
                    data_format='channels_last',
                    name=None):
  """Performs a batch normalization followed by a ReLU.

  Args:
    inputs: `Tensor` of shape `[batch, channels, ...]`.
    is_training_bn: `bool` for whether the model is training.
    relu: `bool` if False, omits the ReLU operation.
    init_zero: `bool` if True, initializes scale parameter of batch
        normalization with 0 instead of 1 (default).
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
    name: the name of the batch normalization layer

  Returns:
    A normalized `Tensor` with the same `data_format`.
  """
  if init_zero:
    gamma_initializer = tf.zeros_initializer()
  else:
    gamma_initializer = tf.ones_initializer()

  if data_format == 'channels_first':
    axis = 1
  else:
    axis = 3

  inputs = tf.layers.batch_normalization(
      inputs=inputs,
      axis=axis,
      momentum=_BATCH_NORM_DECAY,
      epsilon=_BATCH_NORM_EPSILON,
      center=True,
      scale=True,
      training=is_training_bn,
      fused=True,
      gamma_initializer=gamma_initializer,
      name=name,
      trainable=False)

  if relu:
    inputs = tf.nn.relu(inputs)
  return inputs


def fixed_padding(inputs, kernel_size, data_format='channels_last'):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]` or
        `[batch, height, width, channels]` depending on `data_format`.
    kernel_size: `int` kernel size to be used for `conv2d` or max_pool2d`
        operations. Should be a positive integer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A padded `Tensor` of the same `data_format` with size either intact
    (if `kernel_size == 1`) or padded (if `kernel_size > 1`).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  if data_format == 'channels_first':
    padded_inputs = tf.pad(
        inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(
        inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

  return padded_inputs


def conv2d_fixed_padding(inputs,
                         filters,
                         kernel_size,
                         strides,
                         data_format='channels_last',
                         trainable=True):
  """Strided 2-D convolution with explicit padding.

  The padding is consistent and is based only on `kernel_size`, not on the
  dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

  Args:
    inputs: `Tensor` of size `[batch, channels, height_in, width_in]`.
    filters: `int` number of filters in the convolution.
    kernel_size: `int` size of the kernel to be used in the convolution.
    strides: `int` strides of the convolution.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A `Tensor` of shape `[batch, filters, height_out, width_out]`.
  """
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format=data_format)

  return tf.layers.conv2d(
      inputs=inputs,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'),
      use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format,
      trainable=trainable)


def residual_block(inputs,
                   filters,
                   is_training_bn,
                   strides,
                   use_projection=False,
                   data_format='channels_last', training=True, trainable=True):
  """Standard building block for residual networks with BN after convolutions.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
    is_training_bn: `bool` for whether the model is in training.
    strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
    use_projection: `bool` for whether this block should use a projection
        shortcut (versus the default identity shortcut). This is usually `True`
        for the first block of a block group, which may change the number of
        filters and the resolution.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    The output `Tensor` of the block.
  """
  shortcut = inputs
  if use_projection:
    # Projection shortcut in first layer to match filters and strides
    shortcut = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=1,
        strides=strides,
        data_format=data_format, trainable=trainable)
    shortcut = batch_norm_relu(
        shortcut, is_training_bn, relu=False, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format, trainable=trainable)
  inputs = batch_norm_relu(inputs, is_training_bn, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=1,
      data_format=data_format, trainable=trainable)
  inputs = batch_norm_relu(
      inputs,
      is_training_bn,
      relu=False,
      init_zero=True,
      data_format=data_format)

  return tf.nn.relu(inputs + shortcut)


def bottleneck_block(inputs,
                     filters,
                     is_training_bn,
                     strides,
                     use_projection=False,
                     data_format='channels_last', training=True, trainable=True):
  """Bottleneck block variant for residual networks with BN after convolutions.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
    is_training_bn: `bool` for whether the model is in training.
    strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
    use_projection: `bool` for whether this block should use a projection
        shortcut (versus the default identity shortcut). This is usually `True`
        for the first block of a block group, which may change the number of
        filters and the resolution.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    The output `Tensor` of the block.
  """
  shortcut = inputs
  if use_projection:
    # Projection shortcut only in first block within a group. Bottleneck blocks
    # end with 4 times the number of filters.
    filters_out = 4 * filters
    shortcut = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters_out,
        kernel_size=1,
        strides=strides,
        data_format=data_format, trainable=trainable)
    shortcut = batch_norm_relu(
        shortcut, is_training_bn, relu=False, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=1,
      strides=1,
      data_format=data_format, trainable=trainable)
  inputs = batch_norm_relu(inputs, is_training_bn, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format, trainable=trainable)
  inputs = batch_norm_relu(inputs, is_training_bn, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=4 * filters,
      kernel_size=1,
      strides=1,
      data_format=data_format, trainable=trainable)
  inputs = batch_norm_relu(
      inputs,
      is_training_bn,
      relu=False,
      init_zero=True,
      data_format=data_format)

  return tf.nn.relu(inputs + shortcut)


def block_group(inputs,
                filters,
                block_fn,
                blocks,
                strides,
                is_training_bn,
                name,
                data_format='channels_last', trainable=True):
  """Creates one group of blocks for the ResNet model.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first convolution of the layer.
    block_fn: `function` for the block to use within the model
    blocks: `int` number of blocks contained in the layer.
    strides: `int` stride to use for the first convolution of the layer. If
        greater than 1, this layer will downsample the input.
    is_training_bn: `bool` for whether the model is training.
    name: `str`name for the Tensor output of the block layer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    The output `Tensor` of the block layer.
  """
  # Only the first block per block_group uses projection shortcut and strides.
  inputs = block_fn(
      inputs,
      filters,
      is_training_bn,
      strides,
      use_projection=True,
      data_format=data_format, trainable=trainable)

  for _ in range(1, blocks):
    inputs = block_fn(
        inputs, filters, is_training_bn, 1, data_format=data_format, trainable=trainable)

  return tf.identity(inputs, name)


from config import cfg
from viz import draw_on_img, draw_on_img_with_color, draw_heatmap, draw_boxes


def resnet_v1_c4_backbone(inputs, resnet_depth=50, is_training=False, data_format='channels_last', reuse=None):
    """Returns the ResNet model for a given size and number of output classes."""
    model_params = {
        18: {'block': residual_block, 'layers': [2, 2, 2, 2]},
        34: {'block': residual_block, 'layers': [3, 4, 6, 3]},
        50: {'block': bottleneck_block, 'layers': [3, 4, 6, 3]},
        101: {'block': bottleneck_block, 'layers': [3, 4, 23, 3]},
        152: {'block': bottleneck_block, 'layers': [3, 8, 36, 3]},
        200: {'block': bottleneck_block, 'layers': [3, 24, 36, 3]}
    }

    if resnet_depth not in model_params:
        raise ValueError('Not a valid resnet_depth:', resnet_depth)

    params = model_params[resnet_depth]
    is_training_bn = is_training
    block_fn = params['block']
    layers = params['layers']

    """Creation of the model graph."""
    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=64,
        kernel_size=7,
        strides=2,
        data_format=data_format, trainable=False)
    inputs = tf.identity(inputs, 'initial_conv')
    inputs = batch_norm_relu(inputs, is_training_bn, data_format=data_format)

    inputs = tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=3,
        strides=2,
        padding='SAME',
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_max_pool')

    c2 = block_group(
        inputs=inputs,
        filters=64,
        blocks=layers[0],
        strides=1,
        block_fn=block_fn,
        is_training_bn=is_training_bn,
        name='block_group1',
        data_format=data_format)
    c3 = block_group(
        inputs=c2,
        filters=128,
        blocks=layers[1],
        strides=2,
        block_fn=block_fn,
        is_training_bn=is_training_bn,
        name='block_group2',
        data_format=data_format)
    c4 = block_group(
        inputs=c3,
        filters=256,
        blocks=layers[2],
        strides=2,
        block_fn=block_fn,
        is_training_bn=is_training_bn,
        name='block_group3',
        data_format=data_format)

    if cfg.FRCNN.VISUALIZATION:
        c234 = [c2, c3, c4]
        print('visualization')
        with tf.device('cpu:0'):
            with tf.name_scope('vis_featuremaps'):
                for level in range(len(c234)):
                    feature_map = c234[level]
                    draw_heatmap(feature_map, name='C{}/heatmap'.format(level + 2),
                                 data_format=data_format)

    return c4


def resnet_v1_c5(inputs, resnet_depth=50, is_training=False, data_format='channels_last', reuse=None):
    """Returns the ResNet model for a given size and number of output classes."""
    model_params = {
        18: {'block': residual_block, 'layers': [2, 2, 2, 2]},
        34: {'block': residual_block, 'layers': [3, 4, 6, 3]},
        50: {'block': bottleneck_block, 'layers': [3, 4, 6, 3]},
        101: {'block': bottleneck_block, 'layers': [3, 4, 23, 3]},
        152: {'block': bottleneck_block, 'layers': [3, 8, 36, 3]},
        200: {'block': bottleneck_block, 'layers': [3, 24, 36, 3]}
    }

    if resnet_depth not in model_params:
        raise ValueError('Not a valid resnet_depth:', resnet_depth)

    params = model_params[resnet_depth]
    is_training_bn = is_training
    block_fn = params['block']
    layers = params['layers']

    c5 = block_group(
            inputs=inputs,
            filters=512,
            blocks=layers[3],
            strides=2,
            block_fn=block_fn,
            is_training_bn=is_training_bn,
            name='block_group4',
            data_format=data_format)

    return c5


def nearest_upsampling(inputs, scale, data_format='channels_last'):
    with tf.name_scope('nearest_upsampling'):
        shape4d = tf.shape(inputs)
        if data_format == 'channels_first':
            bs, c, h, w = shape4d[0], shape4d[1], shape4d[2], shape4d[3]
            bs = -1 if bs is None else bs
            inputs = tf.reshape(inputs, [bs, c, h, 1, w, 1]) * tf.ones([1, 1, 1, scale, 1, scale], dtype=inputs.dtype)
            return tf.reshape(inputs, [bs, c, h * scale, w * scale])
        else:
            bs, h, w, c = shape4d[0], shape4d[1], shape4d[2], shape4d[3]
            bs = -1 if bs is None else bs
            inputs = tf.reshape(inputs, [bs, h, 1, w, 1, c]) * tf.ones([1, 1, scale, 1, scale, 1], dtype=inputs.dtype)
            return tf.reshape(inputs, [bs, h * scale, w * scale, c])


def resnet_v1_fpn_backbone(inputs, resnet_depth=50, is_training=False, data_format='channels_last',
                           reuse=None):
    model_params = {
        18: {'block': residual_block, 'layers': [2, 2, 2, 2]},
        34: {'block': residual_block, 'layers': [3, 4, 6, 3]},
        50: {'block': bottleneck_block, 'layers': [3, 4, 6, 3]},
        101: {'block': bottleneck_block, 'layers': [3, 4, 23, 3]},
        152: {'block': bottleneck_block, 'layers': [3, 8, 36, 3]},
        200: {'block': bottleneck_block, 'layers': [3, 24, 36, 3]}
    }

    params = model_params[resnet_depth]
    is_training_bn = is_training
    block_fn = params['block']
    layers = params['layers']

    """Creation of the model graph."""
    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=64,
        kernel_size=7,
        strides=2,
        data_format=data_format, trainable=False)
    inputs = tf.identity(inputs, 'initial_conv')
    inputs = batch_norm_relu(inputs, is_training_bn, data_format=data_format)

    inputs = tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=3,
        strides=2,
        padding='SAME',
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_max_pool')

    c2 = block_group(
        inputs=inputs,
        filters=64,
        blocks=layers[0],
        strides=1,
        block_fn=block_fn,
        is_training_bn=is_training_bn,
        name='block_group1',
        data_format=data_format)
    c3 = block_group(
        inputs=c2,
        filters=128,
        blocks=layers[1],
        strides=2,
        block_fn=block_fn,
        is_training_bn=is_training_bn,
        name='block_group2',
        data_format=data_format)
    c4 = block_group(
        inputs=c3,
        filters=256,
        blocks=layers[2],
        strides=2,
        block_fn=block_fn,
        is_training_bn=is_training_bn,
        name='block_group3',
        data_format=data_format)

    c5 = block_group(
        inputs=c4,
        filters=512,
        blocks=layers[3],
        strides=2,
        block_fn=block_fn,
        is_training_bn=is_training_bn,
        name='block_group4',
        data_format=data_format)

    c2345 = [c2, c3, c4, c5]

    kernel_init = tf.variance_scaling_initializer(scale=1., distribution='uniform')

    with tf.variable_scope('fpn'):
        lat2345 = [tf.layers.conv2d(c, filters=256, kernel_size=1, padding='same', name='level%d'%i,
                                    data_format=data_format, kernel_initializer=kernel_init)
                   for i, c in enumerate(c2345)]  # only c2, c3, c4, c5

        lat_sum_5432 = []
        for idx, lat in enumerate(lat2345[::-1]):
            if idx == 0:
                lat_sum_5432.append(lat)
            else:
                lat = lat + nearest_upsampling(lat_sum_5432[-1], scale=2, data_format=data_format)
                lat_sum_5432.append(lat)

        p2345 = [tf.layers.conv2d(c, filters=256, strides=1, kernel_size=3, padding='same', name='p%d'%(i+2),
                                  data_format=data_format, kernel_initializer=kernel_init)
                 for i, c in enumerate(lat_sum_5432[::-1])]
        p6 = tf.layers.conv2d(p2345[-1], filters=256, strides=2, kernel_size=3, padding='same', name='p6',
                              data_format=data_format, kernel_initializer=kernel_init, activation=tf.nn.relu)

        p23456 = p2345 + [p6]
        # for level in range(len(p34567)):
        #     axis = 3 if data_format == 'channels_last' else 1
        #     p34567[level] = tf.layers.batch_normalization(p34567[level],
        #                                                   axis=axis,  momentum=cfg.BACKBONE.BATCH_NORM_DECAY,
        #                                                   epsilon=cfg.BACKBONE.BATCH_NORM_EPSILON,
        #                                                   center=True, scale=True,  training=is_training, fused=True,
        #                                                   name='p%d-bn'%(level+3))

    if cfg.FCOS.VISUALIZATION:
        print('visualization ')
        with tf.device('cpu:0'):
            for level in range(len(c2345)):
                feature_map = c2345[level]
                draw_heatmap(feature_map, name='C{}/heatmap'.format(level + 2),
                             data_format=cfg.BACKBONE.DATA_FORMAT)
            for level in range(len(p23456)):
                feature_map = p23456[level]
                draw_heatmap(feature_map, name='P{}/heatmap'.format(level + 2),
                             data_format=cfg.BACKBONE.DATA_FORMAT)

    return p23456


def resnet_v1_retinanet_backbone(inputs, resnet_depth=50, is_training=False, data_format='channels_last'):
    model_params = {
      18: {'block': residual_block, 'layers': [2, 2, 2, 2]},
      34: {'block': residual_block, 'layers': [3, 4, 6, 3]},
      50: {'block': bottleneck_block, 'layers': [3, 4, 6, 3]},
      101: {'block': bottleneck_block, 'layers': [3, 4, 23, 3]},
      152: {'block': bottleneck_block, 'layers': [3, 8, 36, 3]},
      200: {'block': bottleneck_block, 'layers': [3, 24, 36, 3]}
    }

    params = model_params[resnet_depth]
    is_training_bn = is_training
    block_fn = params['block']
    layers = params['layers']

    """Creation of the model graph."""
    inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=64,
      kernel_size=7,
      strides=2,
      data_format=data_format, trainable=False)
    inputs = tf.identity(inputs, 'initial_conv')
    inputs = batch_norm_relu(inputs, is_training_bn, data_format=data_format)

    inputs = tf.layers.max_pooling2d(
      inputs=inputs,
      pool_size=3,
      strides=2,
      padding='SAME',
      data_format=data_format)
    inputs = tf.identity(inputs, 'initial_max_pool')

    c2 = block_group(
      inputs=inputs,
      filters=64,
      blocks=layers[0],
      strides=1,
      block_fn=block_fn,
      is_training_bn=is_training_bn,
      name='block_group1',
      data_format=data_format)
    c3 = block_group(
      inputs=c2,
      filters=128,
      blocks=layers[1],
      strides=2,
      block_fn=block_fn,
      is_training_bn=is_training_bn,
      name='block_group2',
      data_format=data_format)
    c4 = block_group(
      inputs=c3,
      filters=256,
      blocks=layers[2],
      strides=2,
      block_fn=block_fn,
      is_training_bn=is_training_bn,
      name='block_group3',
      data_format=data_format)

    c5 = block_group(
      inputs=c4,
      filters=512,
      blocks=layers[3],
      strides=2,
      block_fn=block_fn,
      is_training_bn=is_training_bn,
      name='block_group4',
      data_format=data_format)

    c345 = [c3, c4, c5]

    kernel_init = tf.variance_scaling_initializer(scale=1., distribution='uniform')

    with tf.variable_scope('fpn'):
        lat345 = [tf.layers.conv2d(c, filters=256, kernel_size=1, padding='same', name='level%d'%i,
                                   data_format=data_format, kernel_initializer=kernel_init)
                  for i, c in enumerate(c345)]  # only c3, c4, c5

        lat_sum_543 = []
        for idx, lat in enumerate(lat345[::-1]):
            if idx == 0:
                lat_sum_543.append(lat)
            else:
                lat = lat + nearest_upsampling(lat_sum_543[-1], scale=2, data_format=data_format)
                lat_sum_543.append(lat)

        p345 = [tf.layers.conv2d(c, filters=256, strides=1, kernel_size=3, padding='same', name='p%d'%(i+3),
                                 data_format=data_format, kernel_initializer=kernel_init)
                for i, c in enumerate(lat_sum_543[::-1])]
        p6 = tf.layers.conv2d(p345[-1], filters=256, strides=2, kernel_size=3, padding='same', name='p6',
                              data_format=data_format, kernel_initializer=kernel_init, activation=tf.nn.relu)
        p7 = tf.layers.conv2d(p6, filters=256, strides=2, kernel_size=3, padding='same', name='p7',
                              data_format=data_format, kernel_initializer=kernel_init)

        p34567 = p345 + [p6, p7]
        # for level in range(len(p34567)):
        #     axis = 3 if data_format == 'channels_last' else 1
        #     p34567[level] = tf.layers.batch_normalization(p34567[level],
        #                                                   axis=axis,  momentum=cfg.BACKBONE.BATCH_NORM_DECAY,
        #                                                   epsilon=cfg.BACKBONE.BATCH_NORM_EPSILON,
        #                                                   center=True, scale=True,  training=is_training, fused=True,
        #                                                   name='p%d-bn'%(level+3))

    if cfg.VISUALIZATION:
        print('visualization ')
        with tf.device('cpu:0'):
            for level in range(len(c345)):
                feature_map = c345[level]
                draw_heatmap(feature_map, name='C{}/heatmap'.format(level + 3),
                             data_format=cfg.BACKBONE.DATA_FORMAT)
            for level in range(len(p34567)):
                feature_map = p34567[level]
                draw_heatmap(feature_map, name='P{}/heatmap'.format(level + 3),
                             data_format=cfg.BACKBONE.DATA_FORMAT)

    return p34567


def resnet_v1_fcos_backbone(inputs, resnet_depth=50, is_training=False, data_format='channels_last'):
    model_params = {
      18: {'block': residual_block, 'layers': [2, 2, 2, 2]},
      34: {'block': residual_block, 'layers': [3, 4, 6, 3]},
      50: {'block': bottleneck_block, 'layers': [3, 4, 6, 3]},
      101: {'block': bottleneck_block, 'layers': [3, 4, 23, 3]},
      152: {'block': bottleneck_block, 'layers': [3, 8, 36, 3]},
      200: {'block': bottleneck_block, 'layers': [3, 24, 36, 3]}
    }

    params = model_params[resnet_depth]
    is_training_bn = is_training
    block_fn = params['block']
    layers = params['layers']

    """Creation of the model graph."""
    inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=64,
      kernel_size=7,
      strides=2,
      data_format=data_format, trainable=False)
    inputs = tf.identity(inputs, 'initial_conv')
    inputs = batch_norm_relu(inputs, is_training_bn, data_format=data_format)

    inputs = tf.layers.max_pooling2d(
      inputs=inputs,
      pool_size=3,
      strides=2,
      padding='SAME',
      data_format=data_format)
    inputs = tf.identity(inputs, 'initial_max_pool')

    c2 = block_group(
      inputs=inputs,
      filters=64,
      blocks=layers[0],
      strides=1,
      block_fn=block_fn,
      is_training_bn=is_training_bn,
      name='block_group1',
      data_format=data_format)
    c3 = block_group(
      inputs=c2,
      filters=128,
      blocks=layers[1],
      strides=2,
      block_fn=block_fn,
      is_training_bn=is_training_bn,
      name='block_group2',
      data_format=data_format)
    c4 = block_group(
      inputs=c3,
      filters=256,
      blocks=layers[2],
      strides=2,
      block_fn=block_fn,
      is_training_bn=is_training_bn,
      name='block_group3',
      data_format=data_format)

    c5 = block_group(
      inputs=c4,
      filters=512,
      blocks=layers[3],
      strides=2,
      block_fn=block_fn,
      is_training_bn=is_training_bn,
      name='block_group4',
      data_format=data_format)

    c345 = [c3, c4, c5]

    kernel_init = tf.variance_scaling_initializer(scale=1., distribution='uniform')

    with tf.variable_scope('fpn'):
        lat345 = [tf.layers.conv2d(c, filters=256, kernel_size=1, padding='same', name='level%d'%i,
                                   data_format=data_format, kernel_initializer=kernel_init)
                  for i, c in enumerate(c345)]  # only c3, c4, c5

        lat_sum_543 = []
        for idx, lat in enumerate(lat345[::-1]):
            if idx == 0:
                lat_sum_543.append(lat)
            else:
                lat = lat + nearest_upsampling(lat_sum_543[-1], scale=2, data_format=data_format)
                lat_sum_543.append(lat)

        p345 = [tf.layers.conv2d(c, filters=256, strides=1, kernel_size=3, padding='same', name='p%d'%(i+3),
                                 data_format=data_format, kernel_initializer=kernel_init)
                for i, c in enumerate(lat_sum_543[::-1])]
        p6 = tf.layers.conv2d(p345[-1], filters=256, strides=2, kernel_size=3, padding='same', name='p6',
                              data_format=data_format, kernel_initializer=kernel_init, activation=tf.nn.relu)
        p7 = tf.layers.conv2d(p6, filters=256, strides=2, kernel_size=3, padding='same', name='p7',
                              data_format=data_format, kernel_initializer=kernel_init)

        p34567 = p345 + [p6, p7]
        # for level in range(len(p34567)):
        #     axis = 3 if data_format == 'channels_last' else 1
        #     p34567[level] = tf.layers.batch_normalization(p34567[level],
        #                                                   axis=axis,  momentum=cfg.BACKBONE.BATCH_NORM_DECAY,
        #                                                   epsilon=cfg.BACKBONE.BATCH_NORM_EPSILON,
        #                                                   center=True, scale=True,  training=is_training, fused=True,
        #                                                   name='p%d-bn'%(level+3))

    if cfg.FCOS.VISUALIZATION:
        print('visualization ')
        with tf.device('cpu:0'):
            for level in range(len(c345)):
                feature_map = c345[level]
                draw_heatmap(feature_map, name='C{}/heatmap'.format(level + 3),
                             data_format=cfg.BACKBONE.DATA_FORMAT)
            for level in range(len(p34567)):
                feature_map = p34567[level]
                draw_heatmap(feature_map, name='P{}/heatmap'.format(level + 3),
                             data_format=cfg.BACKBONE.DATA_FORMAT)

    return p34567


# TODO(b/111271774): Removes this wrapper once b/111271774 is resolved.
def resize_bilinear(images, size, output_type):
  """Returns resized images as output_type.

  Args:
    images: A tensor of size [batch, height_in, width_in, channels].
    size: A 1-D int32 Tensor of 2 elements: new_height, new_width. The new size
      for the images.
    output_type: The destination type.
  Returns:
    A tensor of size [batch, height_out, width_out, channels] as a dtype of
      output_type.
  """
  images = tf.image.resize_bilinear(images, size, align_corners=True)
  return tf.cast(images, output_type)


## RetinaNet specific layers
def class_net(images, level, num_classes, num_anchors=6, is_training_bn=False):
  """Class prediction network for RetinaNet."""
  for i in range(4):
    images = tf.layers.conv2d(
        images,
        256,
        kernel_size=(3, 3),
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        activation=None,
        padding='same',
        name='class-%d' % i)
    # The convolution layers in the class net are shared among all levels, but
    # each level has its batch normlization to capture the statistical
    # difference among different levels.
    images = batch_norm_relu(images, is_training_bn, relu=True, init_zero=False,
                             name='class-%d-bn-%d' % (i, level))

  classes = tf.layers.conv2d(
      images,
      num_classes * num_anchors,
      kernel_size=(3, 3),
      bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
      padding='same',
      name='class-predict')

  return classes


def box_net(images, level, num_anchors=6, is_training_bn=False):
  """Box regression network for RetinaNet."""
  for i in range(4):
    images = tf.layers.conv2d(
        images,
        256,
        kernel_size=(3, 3),
        activation=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        padding='same',
        name='box-%d' % i)
    # The convolution layers in the box net are shared among all levels, but
    # each level has its batch normlization to capture the statistical
    # difference among different levels.
    images = batch_norm_relu(images, is_training_bn, relu=True, init_zero=False,
                             name='box-%d-bn-%d' % (i, level))

  boxes = tf.layers.conv2d(
      images,
      4 * num_anchors,
      kernel_size=(3, 3),
      bias_initializer=tf.zeros_initializer(),
      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
      padding='same',
      name='box-predict')

  return boxes


def resnet_fpn(features,
               min_level=3,
               max_level=7,
               resnet_depth=50,
               is_training_bn=False,
               use_nearest_upsampling=True):
  """ResNet feature pyramid networks."""
  # upward layers
  with tf.variable_scope('resnet%s' % resnet_depth):
    resnet_fn = resnet_v1(resnet_depth)
    u2, u3, u4, u5 = resnet_fn(features, is_training_bn)

  feats_bottom_up = {
      2: u2,
      3: u3,
      4: u4,
      5: u5,
  }

  with tf.variable_scope('resnet_fpn'):
    # lateral connections
    feats_lateral = {}
    for level in range(min_level, _RESNET_MAX_LEVEL + 1):
      feats_lateral[level] = tf.layers.conv2d(
          feats_bottom_up[level],
          filters=256,
          kernel_size=(1, 1),
          padding='same',
          name='l%d' % level)

    # add top-down path
    feats = {_RESNET_MAX_LEVEL: feats_lateral[_RESNET_MAX_LEVEL]}
    for level in range(_RESNET_MAX_LEVEL - 1, min_level - 1, -1):
      if use_nearest_upsampling:
        feats[level] = nearest_upsampling(feats[level + 1],
                                          2) + feats_lateral[level]
      else:
        feats[level] = resize_bilinear(
            feats[level + 1], tf.shape(feats_lateral[level])[1:3],
            feats[level + 1].dtype) + feats_lateral[level]

    # add post-hoc 3x3 convolution kernel
    for level in range(min_level, _RESNET_MAX_LEVEL + 1):
      feats[level] = tf.layers.conv2d(
          feats[level],
          filters=256,
          strides=(1, 1),
          kernel_size=(3, 3),
          padding='same',
          name='post_hoc_d%d' % level)

    # coarser FPN levels introduced for RetinaNet
    for level in range(_RESNET_MAX_LEVEL + 1, max_level + 1):
      feats_in = feats[level - 1]
      if level > _RESNET_MAX_LEVEL + 1:
        feats_in = tf.nn.relu(feats_in)
      feats[level] = tf.layers.conv2d(
          feats_in,
          filters=256,
          strides=(2, 2),
          kernel_size=(3, 3),
          padding='same',
          name='p%d' % level)
    # add batchnorm
    for level in range(min_level, max_level + 1):
      feats[level] = tf.layers.batch_normalization(
          inputs=feats[level],
          momentum=_BATCH_NORM_DECAY,
          epsilon=_BATCH_NORM_EPSILON,
          center=True,
          scale=True,
          training=is_training_bn,
          fused=True,
          name='p%d-bn' % level)

  return feats


def retinanet(features,
              min_level=3,
              max_level=7,
              num_classes=90,
              num_anchors=6,
              resnet_depth=50,
              use_nearest_upsampling=True,
              is_training_bn=False):
  """RetinaNet classification and regression model."""
  # create feature pyramid networks
  feats = resnet_fpn(features, min_level, max_level, resnet_depth,
                     is_training_bn, use_nearest_upsampling)
  # add class net and box net in RetinaNet. The class net and the box net are
  # shared among all the levels.
  with tf.variable_scope('retinanet'):
    class_outputs = {}
    box_outputs = {}
    with tf.variable_scope('class_net', reuse=tf.AUTO_REUSE):
      for level in range(min_level, max_level + 1):
        class_outputs[level] = class_net(feats[level], level, num_classes,
                                         num_anchors, is_training_bn)
    with tf.variable_scope('box_net', reuse=tf.AUTO_REUSE):
      for level in range(min_level, max_level + 1):
        box_outputs[level] = box_net(feats[level], level,
                                     num_anchors, is_training_bn)

  return class_outputs, box_outputs


def remove_variables(variables, resnet_depth=50):
  """Removes low-level variables from the input.

  Removing low-level parameters (e.g., initial convolution layer) from training
  usually leads to higher training speed and slightly better testing accuracy.
  The intuition is that the low-level architecture (e.g., ResNet-50) is able to
  capture low-level features such as edges; therefore, it does not need to be
  fine-tuned for the detection task.

  Args:
    variables: all the variables in training
    resnet_depth: the depth of ResNet model

  Returns:
    var_list: a list containing variables for training

  """
  var_list = [v for v in variables
              if v.name.find('resnet%s/conv2d/' % resnet_depth) == -1]
  return var_list


def segmentation_class_net(images,
                           level,
                           num_channels=256,
                           is_training_bn=False):
  """Segmentation Feature Extraction Module.

  Args:
    images: A tensor of size [batch, height_in, width_in, channels_in].
    level: The level of features at FPN output_size /= 2^level.
    num_channels: The number of channels in convolution layers
    is_training_bn: Whether batch_norm layers are in training mode.
  Returns:
    images: A feature tensor of size [batch, output_size, output_size,
      channel_number]
  """

  for i in range(3):
    images = tf.layers.conv2d(
        images,
        num_channels,
        kernel_size=(3, 3),
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        activation=None,
        padding='same',
        name='class-%d' % i)
    images = batch_norm_relu(images, is_training_bn, relu=True, init_zero=False,
                             name='class-%d-bn-%d' % (i, level))
  images = tf.layers.conv2d(
      images,
      num_channels,
      kernel_size=(3, 3),
      bias_initializer=tf.zeros_initializer(),
      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
      activation=None,
      padding='same',
      name='class-final')
  return images


def retinanet_segmentation(features,
                           min_level=3,
                           max_level=5,
                           num_classes=21,
                           resnet_depth=50,
                           use_nearest_upsampling=False,
                           is_training_bn=False):
  """RetinaNet extension for semantic segmentation.

  Args:
    features: A tensor of size [batch, height_in, width_in, channels].
    min_level: The minimum output feature pyramid level. This input defines the
      smallest nominal feature stride = 2^min_level.
    max_level: The maximum output feature pyramid level. This input defines the
      largest nominal feature stride = 2^max_level.
    num_classes: Number of object classes.
    resnet_depth: The depth of ResNet backbone model.
    use_nearest_upsampling: Whether use nearest upsampling for FPN network.
      Alternatively, use bilinear upsampling.
    is_training_bn: Whether batch_norm layers are in training mode.
  Returns:
    A tensor of size [batch, height_l, width_l, num_classes]
      representing pixel-wise predictions before Softmax function.
  """
  feats = resnet_fpn(features, min_level, max_level, resnet_depth,
                     is_training_bn, use_nearest_upsampling)

  with tf.variable_scope('class_net', reuse=tf.AUTO_REUSE):
    for level in range(min_level, max_level + 1):
      feats[level] = segmentation_class_net(
          feats[level], level, is_training_bn=is_training_bn)
      if level == min_level:
        fused_feature = feats[level]
      else:
        if use_nearest_upsampling:
          scale = level / min_level
          feats[level] = nearest_upsampling(feats[level], scale)
        else:
          feats[level] = resize_bilinear(
              feats[level], tf.shape(feats[min_level])[1:3], feats[level].dtype)
        fused_feature += feats[level]
  fused_feature = batch_norm_relu(
      fused_feature, is_training_bn, relu=True, init_zero=False)
  classes = tf.layers.conv2d(
      fused_feature,
      num_classes,
      kernel_size=(3, 3),
      bias_initializer=tf.zeros_initializer(),
      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
      padding='same',
      name='class-predict')

  return classes


