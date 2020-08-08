#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/4/30 20:37
# @Author  : Zisha Zhong
# @Email   : zhongzisha@outlook.com
# @File    : common.py

import re
import logger
import tensorflow as tf
from config import cfg

TF_version = tuple(map(int, tf.__version__.split('.')[:2]))
if TF_version <= (1, 12):
    try:
        from tensorflow.contrib.nccl.python.ops.nccl_ops import \
            _validate_and_load_nccl_so  # deprecated
    except Exception:
        pass
    else:
        _validate_and_load_nccl_so()
    from tensorflow.contrib.nccl.ops import gen_nccl_ops  # deprecated
else:
    from tensorflow.python.ops import gen_nccl_ops


if TF_version <= (1, 12):
    l2_regularizer = tf.contrib.layers.l2_regularizer  # deprecated
    l1_regularizer = tf.contrib.layers.l1_regularizer  # deprecated
else:
    # oh these little dirty details
    l2_regularizer = lambda x: tf.keras.regularizers.l2(x * 0.5)  # noqa
    l1_regularizer = tf.keras.regularizers.l1


def regularize_cost(regex, func, name='regularize_cost'):
    """
    Apply a regularizer on trainable variables matching the regex, and print
    the matched variables (only print once in multi-tower training).
    In replicated mode, it will only regularize variables within the current tower.

    If called under a TowerContext with `is_training==False`, this function returns a zero constant tensor.

    Args:
        regex (str): a regex to match variable names, e.g. "conv.*/W"
        func: the regularization function, which takes a tensor and returns a scalar tensor.
            E.g., ``tf.nn.l2_loss, tf.contrib.layers.l1_regularizer(0.001)``.

    Returns:
        tf.Tensor: a scalar, the total regularization cost.

    Example:
        .. code-block:: python

            cost = cost + regularize_cost("fc.*/W", l2_regularizer(1e-5))
    """
    assert len(regex)
    params = tf.trainable_variables()

    names = []

    with tf.name_scope(name + '_internals'):
        costs = []
        for p in params:
            para_name = p.op.name
            if re.search(regex, para_name):
                regloss = func(p)
                assert regloss.dtype.is_floating, regloss
                # Some variables may not be fp32, but it should
                # be fine to assume regularization in fp32
                if regloss.dtype != tf.float32:
                    regloss = tf.cast(regloss, tf.float32)
                costs.append(regloss)
                names.append(p.name)
        if not costs:
            return tf.constant(0, dtype=tf.float32, name='empty_' + name)
    logger.info("The following tensors will be regularized: {}".format(', '.join(names)))

    return tf.add_n(costs, name=name)


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(values=grads, axis=0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def sum_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    sum_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_sum(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        sum_grads.append(grad_and_var)
    return sum_grads


class GradientClipOptimizer(tf.train.Optimizer):
    def __init__(self, opt, clip_norm):
        self.opt = opt
        self.clip_norm = clip_norm

    def compute_gradients(self, *args, **kwargs):
        return self.opt.compute_gradients(*args, **kwargs)
    """
    def apply_gradients(self, *args, **kwargs):
        return self.opt.apply_gradients(*args, **kwargs)
    """
    def apply_gradients(self, gradvars, global_step=None, name=None):
        old_grads, v = zip(*gradvars)
        all_are_finite = tf.reduce_all([tf.reduce_all(tf.is_finite(g)) for g in old_grads])
        clipped_grads, _ = tf.clip_by_global_norm(old_grads, self.clip_norm,
                                                  use_norm=tf.cond(
                                                      all_are_finite,
                                                      lambda: tf.global_norm(old_grads),
                                                      lambda: tf.constant(self.clip_norm, dtype=tf.float32)), name='clip_by_global_norm')
        gradvars = list(zip(clipped_grads, v))
        return self.opt.apply_gradients(gradvars, global_step, name)

    def get_slot(self, *args, **kwargs):
        return self.opt.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        return self.opt.get_slot_names(*args, **kwargs)

    def variables(self, *args, **kwargs):
        return self.opt.variables(*args, **kwargs)


def warmup_lr_schedule(init_learning_rate, global_step, warmup_step):
    def warmup(end_lr, global_step, warmup_step):
        start_lr = end_lr * 0.1
        global_step = tf.cast(global_step, tf.float32)
        return start_lr + (end_lr - start_lr) * global_step / warmup_step

    def decay(start_lr, global_step):
        boundaries = cfg.TRAIN.LR_BOUNDARIES   # 1x
        values = [start_lr, start_lr*0.1, start_lr*0.01]
        return tf.train.piecewise_constant(global_step, boundaries=boundaries, values=values)

    return tf.cond(tf.less_equal(global_step, warmup_step),
                   lambda: warmup(init_learning_rate, global_step, warmup_step),
                   lambda: decay(init_learning_rate, global_step))

