#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/15 9:32
# @Author  : Zisha Zhong
# @Email   : zhongzisha@outlook.com
# @File    : retinanet_model.py

import math
import numpy as np
import tensorflow as tf
from config import cfg

from tensorflow.python.ops import gen_image_ops

from viz import draw_boxes, draw_boxes_with_scores

from resnet_model import resnet_v1_retinanet_backbone


def area(boxes):
    xmin, ymin, xmax, ymax = \
        tf.split(boxes, num_or_size_splits=4, axis=1)
    return tf.squeeze((xmax - xmin) * (ymax - ymin), axis=[1])


def pairwise_intersection(boxlist1, boxlist2):
    # boxlist1: Nx4
    # boxlist2: Mx4
    xmin1, ymin1, xmax1, ymax1 = \
        tf.split(boxlist1, num_or_size_splits=4, axis=1)
    xmin2, ymin2, xmax2, ymax2 = \
        tf.split(boxlist2, num_or_size_splits=4, axis=1)
    all_pairs_min_ymax = tf.minimum(ymax1, tf.transpose(ymax2))
    all_pairs_max_ymin = tf.maximum(ymin1, tf.transpose(ymin2))
    intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = tf.minimum(xmax1, tf.transpose(xmax2))
    all_pairs_max_xmin = tf.maximum(xmin1, tf.transpose(xmin2))
    intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths


def pairwise_iou(boxlist1, boxlist2):
    intersections = pairwise_intersection(boxlist1, boxlist2)
    areas1 = area(boxlist1)
    areas2 = area(boxlist2)
    unions = tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections
    return tf.where(tf.equal(intersections, 0.0),
                    tf.zeros_like(intersections),
                    tf.truediv(intersections, unions))


# for RetinaNet
def get_all_anchors_retinanet(height, width, stride, sizes, ratios):
    anchors = []
    # for sz in sizes:
    #     for ratio in ratios:
    #         w = np.sqrt(sz * sz / ratio)
    #         h = ratio * w
    #         anchors.append([-w, -h, w, h])
    for sz in sizes:
        for ratio in ratios:
            for scale in cfg.RETINANET.ANCHOR_SCALES:  # this is for RetinaNet only
                w = np.sqrt(sz * sz * scale / ratio)
                h = ratio * w
                anchors.append([-w, -h, w, h])
    cell_anchors = np.asarray(anchors) * 0.5
    feat_w = int(width // stride)
    feat_h = int(height // stride)
    shift_x = np.arange(feat_w) * stride
    shift_y = np.arange(feat_h) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    shifts = np.vstack((shift_x, shift_y, shift_x, shift_y)).transpose()
    K = shifts.shape[0]
    A = cell_anchors.shape[0]
    field_of_anchors = cell_anchors.reshape((1, A, 4)) + \
        shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    field_of_anchors = field_of_anchors.reshape((feat_h, feat_w, A, 4))
    field_of_anchors = field_of_anchors.astype(np.float32)
    return field_of_anchors


def encode_bbox_target(boxes, anchors):
    anchors_x1y1x2y2 = tf.reshape(anchors, (-1, 2, 2))
    anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, num_or_size_splits=2, axis=1)
    waha = anchors_x2y2 - anchors_x1y1
    xaya = (anchors_x2y2 + anchors_x1y1) * 0.5

    boxes_x1y1x2y2 = tf.reshape(boxes, (-1, 2, 2))
    boxes_x1y1, boxes_x2y2 = tf.split(boxes_x1y1x2y2, num_or_size_splits=2, axis=1)
    wbhb = boxes_x2y2 - boxes_x1y1
    xbyb = (boxes_x2y2 + boxes_x1y1) * 0.5

    txty = (xbyb - xaya) / waha
    twth = tf.log(wbhb / waha)
    encoded = tf.concat([txty, twth], axis=1)  # -1x2x2
    return tf.reshape(encoded, tf.shape(boxes))


def encode_bbox_target_batch(boxes, anchors):
    # boxes: BSxfHxfWxNAx4
    # anchors: fHxfWxNAx4
    anchors = tf.tile(tf.expand_dims(anchors, axis=0), [tf.shape(boxes)[0], 1, 1, 1, 1])
    anchors_x1y1x2y2 = tf.reshape(anchors, (-1, 2, 2))
    anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, num_or_size_splits=2, axis=1)
    waha = anchors_x2y2 - anchors_x1y1
    xaya = (anchors_x2y2 + anchors_x1y1) * 0.5

    boxes_x1y1x2y2 = tf.reshape(boxes, (-1, 2, 2))
    boxes_x1y1, boxes_x2y2 = tf.split(boxes_x1y1x2y2, num_or_size_splits=2, axis=1)
    wbhb = boxes_x2y2 - boxes_x1y1
    xbyb = (boxes_x2y2 + boxes_x1y1) * 0.5

    txty = (xbyb - xaya) / waha
    twth = tf.log(wbhb / waha)
    encoded = tf.concat([txty, twth], axis=1)  # -1x2x2
    return tf.reshape(encoded, tf.shape(boxes))


def decode_bbox_target(box_predictions, anchors):
    orig_shape = tf.shape(anchors)
    box_pred_txtytwth = tf.reshape(box_predictions, (-1, 2, 2))
    box_pred_txty, box_pred_twth = tf.split(box_pred_txtytwth, num_or_size_splits=2, axis=1)
    anchors_x1y1x2y2 = tf.reshape(anchors, (-1, 2, 2))
    anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, num_or_size_splits=2, axis=1)

    waha = anchors_x2y2 - anchors_x1y1
    xaya = (anchors_x2y2 + anchors_x1y1) * 0.5

    wbhb = tf.exp(box_pred_twth) * waha
    xbyb = box_pred_txty * waha + xaya
    x1y1 = xbyb - wbhb * 0.5
    x2y2 = xbyb + wbhb * 0.5
    decoded = tf.concat([x1y1, x2y2], axis=1)
    return tf.reshape(decoded, orig_shape)


def decode_bbox_target_numpy(box_predictions, anchors):
    orig_shape = anchors.shape
    box_pred_txtytwth = np.reshape(box_predictions, (-1, 2, 2))
    box_pred_txty, box_pred_twth = np.split(box_pred_txtytwth, 2, axis=1)
    anchors_x1y1x2y2 = np.reshape(anchors, (-1, 2, 2))
    anchors_x1y1, anchors_x2y2 = np.split(anchors_x1y1x2y2, 2, axis=1)

    waha = anchors_x2y2 - anchors_x1y1
    xaya = (anchors_x2y2 + anchors_x1y1) * 0.5

    wbhb = np.exp(box_pred_twth) * waha
    xbyb = box_pred_txty * waha + xaya
    x1y1 = xbyb - wbhb * 0.5
    x2y2 = xbyb + wbhb * 0.5
    decoded = np.concatenate([x1y1, x2y2], axis=1)
    return np.reshape(decoded, orig_shape)


def decode_bbox_target_batch(box_predictions, anchors):
    # box_predictions: BSxfHxfWxNAx4
    # anchors: fHxfWxNAx4
    print('box_predictions', box_predictions)
    anchors = tf.tile(tf.expand_dims(anchors, axis=0), [tf.shape(box_predictions)[0], 1, 1, 1, 1])
    print('anchors', anchors)
    orig_shape = tf.shape(anchors)
    box_pred_txtytwth = tf.reshape(box_predictions, (-1, 2, 2))
    box_pred_txty, box_pred_twth = tf.split(box_pred_txtytwth, num_or_size_splits=2, axis=1)
    anchors_x1y1x2y2 = tf.reshape(anchors, (-1, 2, 2))
    anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, num_or_size_splits=2, axis=1)

    waha = anchors_x2y2 - anchors_x1y1
    xaya = (anchors_x2y2 + anchors_x1y1) * 0.5

    wbhb = tf.exp(box_pred_twth) * waha
    xbyb = box_pred_txty * waha + xaya
    x1y1 = xbyb - wbhb * 0.5
    x2y2 = xbyb + wbhb * 0.5
    decoded = tf.concat([x1y1, x2y2], axis=1)
    return tf.reshape(decoded, orig_shape)


def smooth_l1_loss(box_logits, box_targets, sigma=1.0):
    sigma_2 = sigma ** 2
    box_diff = box_logits - box_targets
    abs_box_diff = tf.abs(box_diff)
    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_box_diff, 1. / sigma_2)))
    loss_box = tf.pow(box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + \
               (abs_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    loss_box = tf.reduce_sum(loss_box)
    return loss_box


def iou_loss(gt_boxes, pred_boxes, loss_type='giou'):
    pred_boxes = tf.cast(pred_boxes, tf.float32)
    gt_boxes = tf.stop_gradient(tf.cast(gt_boxes, tf.float32))

    pred_left, pred_top, pred_right, pred_bottom = tf.split(pred_boxes, num_or_size_splits=4, axis=1)
    gt_left, gt_top, gt_right, gt_bottom = tf.split(gt_boxes, num_or_size_splits=4, axis=1)

    gt_area = (gt_left + gt_right) * (gt_top + gt_bottom)
    pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

    w_intersect = tf.minimum(pred_left, gt_left) + tf.minimum(pred_right, gt_right)
    g_w_intersect = tf.maximum(pred_left, gt_left) + tf.maximum(pred_right, gt_right)
    h_intersect = tf.minimum(pred_bottom, gt_bottom) + tf.minimum(pred_top, gt_top)
    g_h_intersect = tf.maximum(pred_bottom, gt_bottom) + tf.maximum(pred_top, gt_top)

    ac_union = g_w_intersect * g_h_intersect + 1e-7
    area_intersect = w_intersect * h_intersect
    area_union = gt_area + pred_area - area_intersect
    ious = (area_intersect + 1.0) / (area_union + 1.0)
    gious = ious - (ac_union - area_union) / ac_union
    if loss_type == 'iou':
        loss = -tf.math.log(ious)
    elif loss_type == 'linear_iou':
        loss = 1 - ious
    elif loss_type == 'giou':
        loss = 1 - gious
    elif loss_type == 'dice':
        loss = (2 * area_intersect + 1e-5) / (gt_area + pred_area + 1e-5)
    loss = tf.reduce_sum(loss)
    return loss


def sigmoid_focal_loss(labels, logits, gamma=2.0, alpha=0.25):
    labels = tf.stop_gradient(tf.cast(labels, tf.int32))
    logits = tf.cast(logits, tf.float32)
    num_classes = tf.shape(logits)[1]
    p = tf.reshape(tf.sigmoid(logits), [-1, num_classes])
    t = tf.reshape(labels, [-1, 1])
    # class_range = tf.expand_dims(tf.range(1, num_classes + 1), axis=0)
    # pos1 = tf.cast(tf.equal(t, class_range), tf.float32)
    inds = tf.concat([tf.reshape(tf.range(tf.size(t), dtype=tf.int32), [-1, 1]), t - 1], axis=1)
    pos1 = tf.scatter_nd(inds, updates=tf.ones((tf.size(t),)), shape=[tf.size(t), num_classes])
    term1 = tf.pow(1-p, gamma) * tf.math.log(p)
    term2 = tf.pow(p, gamma) * tf.math.log(1-p)
    pos_t = tf.cast(tf.greater_equal(t, 0), tf.float32)
    loss = -pos1 * term1 * alpha - ((1-pos1)*pos_t) * term2 * (1-alpha)
    loss = tf.reduce_sum(loss)
    return loss


def sigmoid_focal_loss2(labels, logits, gamma=2.0, alpha=0.25):
    labels = tf.stop_gradient(tf.cast(labels, tf.int32))
    logits = tf.cast(logits, tf.float32)
    num_classes = tf.shape(logits)[1]
    p = tf.reshape(tf.sigmoid(logits), [-1, num_classes])
    t = tf.reshape(labels, [-1, 1])
    # class_range = tf.expand_dims(tf.range(1, num_classes + 1), axis=0)
    # pos1 = tf.cast(tf.equal(t, class_range), tf.float32)
    inds = tf.concat([tf.reshape(tf.range(tf.size(t), dtype=tf.int32), [-1, 1]), t - 1], axis=1)
    t = tf.scatter_nd(inds, updates=tf.ones((tf.size(t),)), shape=[tf.size(t), num_classes])

    alpha_factor = tf.ones_like(t) * alpha
    alpha_factor = tf.where(tf.equal(t, 1), alpha_factor, 1 - alpha_factor)
    focal_weight = tf.where(tf.equal(t, 1), 1 - p, p)
    focal_weight = alpha_factor * focal_weight ** gamma

    # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
    loss = - t * tf.math.log(p) - (1 - t) * tf.math.log(1 - p)
    loss = focal_weight * loss

    loss = tf.reduce_sum(loss)
    return loss


def sigmoid_focal_loss3(labels, logits, gamma=2.0, alpha=0.25):
    labels = tf.stop_gradient(tf.cast(labels, tf.int32))
    logits = tf.cast(logits, tf.float32)
    num_classes = tf.shape(logits)[1]
    logits = tf.reshape(logits, [-1, num_classes])
    t = tf.reshape(labels, [-1, 1])
    # class_range = tf.expand_dims(tf.range(1, num_classes + 1), axis=0)
    # pos1 = tf.cast(tf.equal(t, class_range), tf.float32)
    inds = tf.concat([tf.reshape(tf.range(tf.size(t), dtype=tf.int32), [-1, 1]), t - 1], axis=1)
    targets = tf.scatter_nd(inds, updates=tf.ones((tf.size(t),)), shape=[tf.size(t), num_classes])

    positive_label_mask = tf.equal(targets, 1.0)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=targets, logits=logits
    )
    probs = tf.sigmoid(logits)
    probs_gt = tf.where(positive_label_mask, probs, 1.0 - probs)
    modulator = tf.math.pow(1.0 - probs_gt, gamma)
    loss = modulator * cross_entropy
    weighted_loss = tf.where(positive_label_mask, alpha * loss, (1.0 - alpha) * loss)
    loss = tf.reduce_sum(weighted_loss)
    return loss


def sigmoid_focal_loss4(labels, logits, gamma=2.0, alpha=0.25):
    labels = tf.stop_gradient(tf.cast(labels, tf.int32))
    logits = tf.cast(logits, tf.float32)
    num_classes = tf.shape(logits)[1]
    logits = tf.reshape(logits, [-1, num_classes])
    t = tf.reshape(labels, [-1, 1])
    # class_range = tf.expand_dims(tf.range(1, num_classes + 1), axis=0)
    # pos1 = tf.cast(tf.equal(t, class_range), tf.float32)
    inds = tf.concat([tf.reshape(tf.range(tf.size(t), dtype=tf.int32), [-1, 1]), t - 1], axis=1)
    targets = tf.scatter_nd(inds, updates=tf.ones((tf.size(t),)), shape=[tf.size(t), num_classes])

    p = tf.sigmoid(logits)
    ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=targets, logits=logits
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return tf.reduce_sum(loss)


def retinanet_head(features, num_channels=256, num_classes=80,
                   is_training=False, data_format='channels_last',
                   reuse=None):

    kernel_init = tf.random_normal_initializer(stddev=0.01)
    bias_init = tf.constant_initializer(-math.log((1 - cfg.FCOS.PRIOR_PROB) / cfg.FCOS.PRIOR_PROB))

    num_anchors = int(len(cfg.RETINANET.ANCHOR_SCALES) * len(cfg.RETINANET.ANCHOR_RATIOS))

    with tf.variable_scope('retinanet', reuse=reuse):
        outputs = []
        for level, feature in enumerate(features):

            with tf.variable_scope('cls_branch', reuse=tf.AUTO_REUSE):
                l = feature
                for i in range(4):
                    l = tf.layers.conv2d(l, filters=num_channels, kernel_size=3, padding='same',
                                         data_format=data_format, use_bias=True, kernel_initializer=kernel_init,
                                         name='conv%d'%i)
                    l = tf.nn.relu(l)
                cls_tower = l

            with tf.variable_scope('reg_branch', reuse=tf.AUTO_REUSE):
                l = feature
                for i in range(4):
                    l = tf.layers.conv2d(l, filters=num_channels, kernel_size=3, padding='same',
                                         data_format=data_format, use_bias=True, kernel_initializer=kernel_init,
                                         name='conv%d'%i)
                    l = tf.nn.relu(l)
                bbox_tower = l

            with tf.variable_scope('output', reuse=tf.AUTO_REUSE):
                cls_logits = tf.layers.conv2d(cls_tower, filters=num_anchors * num_classes, kernel_size=3,
                                              padding='same', name='cls', data_format=data_format,
                                              kernel_initializer=kernel_init, bias_initializer=bias_init)
                bbox_pred = tf.layers.conv2d(bbox_tower, filters=num_anchors * 4, kernel_size=3,
                                             padding='same', name='box', data_format=data_format,
                                             kernel_initializer=kernel_init)
                outputs.append((cls_logits, bbox_pred))

        return outputs


def clip_boxes(boxes, window):
    boxes = tf.maximum(boxes, 0.0)
    m = tf.tile(tf.reverse(window, [0]), [2])
    boxes = tf.minimum(boxes, tf.cast(m, tf.float32))
    return boxes


def output_predictions_per_level(box_logits, box_deltas, anchors, img_height=0, img_width=0):
    # box_logits: NxHxWxAK
    # box_deltas: NxHxWxA4
    # anchors: HxWxAx4

    N = tf.shape(box_logits)[0]
    num_classes = cfg.DATA.NUM_CATEGORY
    anchors = tf.reshape(anchors, [-1, 4])

    box_scores = tf.sigmoid(box_logits)

    def condition(b_id, final_boxes, final_scores, final_labels, final_inds):
        return tf.less(b_id, N)

    def body(b_id, final_boxes, final_scores, final_labels, final_inds):
        box_scores_this = tf.reshape(box_scores[b_id], [-1])  # (HxWxA, K)
        box_deltas_this = tf.reshape(box_deltas[b_id], [-1, 4])  # (HxWxA, 4)

        # filter low score boxes
        valid_inds = tf.reshape(tf.where(tf.greater_equal(box_scores_this,
                                                          cfg.FRCNN.TEST.RESULT_SCORE_THRESH)), [-1])
        valid_scores = tf.gather(box_scores_this, valid_inds)

        # select topk boxes
        topk = tf.minimum(cfg.RETINANET.TEST_PER_LEVEL_NMS_TOPK, tf.size(valid_inds))
        topk_scores, topk_valid_indices_with_classes = tf.nn.top_k(valid_scores, k=topk, sorted=False)
        topk_indices_with_classes = tf.reshape(tf.gather(valid_inds, topk_valid_indices_with_classes), [-1])
        classes = tf.math.mod(topk_indices_with_classes, num_classes)
        topk_indices = tf.math.floordiv(topk_indices_with_classes, num_classes)
        topk_boxes_deltas_this = tf.gather(box_deltas_this, topk_indices)
        anchors_this = tf.gather(anchors, topk_indices)

        topk_boxes = tf.reshape(decode_bbox_target(topk_boxes_deltas_this, anchors_this), [-1, 4])
        topk_boxes = clip_boxes(topk_boxes, (img_height, img_width))  # topkx4

        boxes_this_image = []
        scores_this_image = []
        labels_this_image = []

        for c in range(num_classes):
            inds = tf.reshape(tf.where(tf.equal(classes, c)), [-1])
            perclass_boxes = tf.reshape(tf.gather(topk_boxes, inds), [-1, 4])
            perclass_scores = tf.reshape(tf.gather(topk_scores, inds), [-1])

            keep = tf.image.non_max_suppression(perclass_boxes, perclass_scores,
                                                max_output_size=cfg.FRCNN.TEST.RESULTS_PER_IM,
                                                iou_threshold=cfg.FRCNN.TEST.NMS_THRESH,
                                                score_threshold=cfg.FRCNN.TEST.RESULT_SCORE_THRESH)
            perclass_boxes = tf.gather(perclass_boxes, keep)
            perclass_scores = tf.gather(perclass_scores, keep)

            boxes_this_image.append(perclass_boxes)
            scores_this_image.append(perclass_scores)
            labels_this_image.append((c+1) * tf.ones_like(perclass_scores, dtype=tf.int64))

        boxes_this_image = tf.concat(boxes_this_image, axis=0)
        scores_this_image = tf.concat(scores_this_image, axis=0)
        labels_this_image = tf.concat(labels_this_image, axis=0)
        inds_this_image = b_id * tf.ones((tf.shape(boxes_this_image)[0],), dtype=tf.int32)

        final_boxes = tf.cond(tf.equal(b_id, 0),
                              lambda: boxes_this_image,
                              lambda: tf.concat([final_boxes, boxes_this_image], axis=0))
        final_scores = tf.cond(tf.equal(b_id, 0),
                               lambda: scores_this_image,
                               lambda: tf.concat([final_scores, scores_this_image], axis=0))
        final_labels = tf.cond(tf.equal(b_id, 0),
                               lambda: labels_this_image,
                               lambda: tf.concat([final_labels, labels_this_image], axis=0))
        final_inds = tf.cond(tf.equal(b_id, 0),
                             lambda: inds_this_image,
                             lambda: tf.concat([final_inds, inds_this_image], axis=0))
        return tf.add(b_id, 1), final_boxes, final_scores, final_labels, final_inds

    b_id = tf.constant(0, dtype=tf.int32)
    final_boxes = tf.zeros([0, 4], dtype=tf.float32)
    final_scores = tf.zeros([0, ], dtype=tf.float32)
    final_labels = tf.zeros([0, ], dtype=tf.int64)
    final_inds = tf.zeros([0, ], dtype=tf.int32)
    index_results = (b_id, final_boxes, final_scores, final_labels, final_inds)
    _, final_boxes, final_scores, final_labels, final_inds = \
        tf.while_loop(condition, body, index_results,
                      shape_invariants=(b_id.get_shape(),
                                        tf.TensorShape([None, 4]),
                                        tf.TensorShape([None]),
                                        tf.TensorShape([None]),
                                        tf.TensorShape([None])))

    return final_boxes, final_scores, final_labels, final_inds


def model_retinanet(inputs, is_training=False, reuse=False,
                    data_format='channels_last', mode='train'):

    if mode == 'train':
        images, gt_boxes, gt_labels, orig_gt_counts, fpn_all_anchors, \
        fpn_anchor_gt_labels, fpn_anchor_gt_boxes = inputs
        # fpn_all_anchors is a list
        # fpn_anchor_gt_labels is a list
        # fpn_anchor_gt_boxes is a list

        ###################################################################################
        # visualization
        if cfg.VISUALIZATION:
            print('visualization ')
            with tf.device('/cpu:0'):
                with tf.name_scope('vis_gt'):
                    images_show = tf.identity(images)
                    image_mean = tf.constant(cfg.PREPROC.PIXEL_MEAN, dtype=tf.float32)  # RGB
                    image_invstd = tf.constant(1.0 / np.asarray(cfg.PREPROC.PIXEL_STD), dtype=tf.float32)
                    images_show /= image_invstd
                    images_show = images_show + image_mean
                    images_show = tf.clip_by_value(images_show * cfg.PREPROC.PIXEL_SCALE, 0, 255)
                    images_show = tf.cast(images_show, tf.uint8)

                    for b_id in range(cfg.TRAIN.BATCH_SIZE_PER_GPU):
                        cur_image = images_show[b_id]
                        cur_gt_boxes = tf.reshape(gt_boxes[b_id], [-1, 4])
                        cur_gt_labels = tf.reshape(gt_labels[b_id], [-1])
                        cur_gt_count = tf.reshape(orig_gt_counts[b_id], [])
                        cur_gt_valid_inds = tf.reshape(tf.range(cur_gt_count, dtype=tf.int32), [-1])
                        cur_gt_boxes = tf.gather(cur_gt_boxes, cur_gt_valid_inds)
                        cur_gt_labels = tf.gather(cur_gt_labels, cur_gt_valid_inds)
                        gt_img = tf.py_func(draw_boxes, inp=[cur_image, cur_gt_boxes, cur_gt_labels,
                                                             'gt_{}'.format(b_id)],
                                            Tout=[tf.uint8])
                        tf.summary.image('gt_img_{}'.format(b_id), gt_img)
        ###################################################################################
    else:
        images = inputs

    if data_format == 'channels_first':
        images = tf.transpose(images, [0, 3, 1, 2])  # NHWC --> NCHW

    features = resnet_v1_retinanet_backbone(inputs=images, resnet_depth=50, is_training=is_training,
                                            data_format=data_format)
    # features: p3, p4, p5, p6, p7

    img_shape = tf.shape(images)
    if data_format == 'channels_first':
        height, width = img_shape[2], img_shape[3]
    else:
        height, width = img_shape[1], img_shape[2]

    if mode != 'train':
        fpn_all_anchors = []
        for idx, (stride, size) in enumerate(zip(cfg.RETINANET.ANCHOR_STRIDES, cfg.RETINANET.ANCHOR_SIZES)):
            feature_shape_this_level = features[idx].get_shape().as_list()
            if data_format == 'channels_first':
                feat_height, feat_width = feature_shape_this_level[2], feature_shape_this_level[3]
            else:
                feat_height, feat_width = feature_shape_this_level[1], feature_shape_this_level[2]

            sizes = (size, )
            num_anchors = int(len(sizes) *
                              len(cfg.RETINANET.ANCHOR_RATIOS) *
                              len(cfg.RETINANET.ANCHOR_SCALES))

            anchors_this_level = \
                tf.py_func(get_all_anchors_retinanet,
                           inp=[height, width, stride, sizes, cfg.RETINANET.ANCHOR_RATIOS],
                           Tout=tf.float32)
            anchors_this_level.set_shape([feat_height, feat_width, num_anchors, 4])

            fpn_all_anchors.append(anchors_this_level)

    if mode == 'train':
        ###################################################################################
        # visualization
        if cfg.VISUALIZATION:
            print('visualization ')
            with tf.device('/cpu:0'):
                with tf.name_scope('vis_anchors'):
                    if data_format == 'channels_first':
                        images_show = tf.identity(tf.transpose(images, [0, 2, 3, 1]))  # NCHW --> NHWC
                    else:
                        images_show = tf.identity(images)
                    image_mean = tf.constant(cfg.PREPROC.PIXEL_MEAN, dtype=tf.float32)  # RGB
                    image_invstd = tf.constant(1.0 / np.asarray(cfg.PREPROC.PIXEL_STD), dtype=tf.float32)
                    images_show /= image_invstd
                    images_show = images_show + image_mean
                    images_show = tf.clip_by_value(images_show * cfg.PREPROC.PIXEL_SCALE, 0, 255)
                    images_show = tf.cast(images_show, tf.uint8)

                    for idx in range(len(features)):
                        with tf.name_scope('Level-{}'.format(3 + idx)):
                            all_anchors_for_show = tf.reshape(fpn_all_anchors[idx], [-1, 4])

                            for b_id in range(cfg.TRAIN.BATCH_SIZE_PER_GPU):
                                cur_image = images_show[b_id]
                                cur_anchor_labels = tf.reshape(fpn_anchor_gt_labels[idx][b_id], [-1])
                                pos_inds = tf.reshape(tf.where(tf.greater(cur_anchor_labels, 0)), [-1])
                                neg_inds = tf.reshape(tf.where(tf.equal(cur_anchor_labels, 0)), [-1])
                                ignore_inds = tf.reshape(
                                    tf.where(tf.equal(cur_anchor_labels, -1)), [-1]
                                )

                                pos_anchor_boxes = tf.gather(all_anchors_for_show, pos_inds)
                                neg_anchor_boxes = tf.gather(all_anchors_for_show, neg_inds)
                                ignore_anchor_boxes = tf.gather(all_anchors_for_show, ignore_inds)

                                pos_anchor_labels = tf.gather(cur_anchor_labels, pos_inds)
                                neg_anchor_labels = tf.gather(cur_anchor_labels, neg_inds)
                                ignore_anchor_labels = tf.gather(cur_anchor_labels, ignore_inds)

                                pos_img = tf.py_func(draw_boxes,
                                                     inp=[cur_image, pos_anchor_boxes, pos_anchor_labels,
                                                          'pos_anchor_{}_{}'.format(idx+3, b_id)],
                                                     Tout=[tf.uint8])
                                tf.summary.image('pos_anchor_img_{}'.format(b_id), pos_img)

                                neg_img = tf.py_func(draw_boxes,
                                                     inp=[cur_image, neg_anchor_boxes, neg_anchor_labels,
                                                          'neg_anchor_{}_{}'.format(idx+3, b_id)],
                                                     Tout=[tf.uint8])
                                tf.summary.image('neg_anchor_img_{}'.format(b_id), neg_img)

                                ignore_img = tf.py_func(draw_boxes,
                                                        inp=[cur_image, ignore_anchor_boxes, ignore_anchor_labels,
                                                          'ignore_anchor_{}_{}'.format(idx + 3, b_id)],
                                                        Tout=[tf.uint8])
                                tf.summary.image('ignore_anchor_img_{}'.format(b_id), ignore_img)
        ###################################################################################

    retinanet_outputs = retinanet_head(features, num_channels=256, num_classes=cfg.DATA.NUM_CATEGORY,
                                       is_training=is_training, data_format=data_format, reuse=reuse)

    if mode == 'train':
        # class, bbox_reg
        cls_logits_flatten = []
        reg_logits_flatten = []
        cls_labels_flatten = []
        reg_targets_flatten = []
        for l in range(len(cfg.RETINANET.ANCHOR_STRIDES)):
            cls_logits_flatten.append(tf.reshape(tf.transpose(retinanet_outputs[l][0], [0, 2, 3, 1]),
                                              [-1, cfg.DATA.NUM_CATEGORY]))
            reg_logits_flatten.append(tf.reshape(tf.transpose(retinanet_outputs[l][1], [0, 2, 3, 1]),
                                              [-1, 4]))
            cls_labels_flatten.append(tf.reshape(fpn_anchor_gt_labels[l], [-1]))

            anchor_gt_boxes_encoded = encode_bbox_target_batch(fpn_anchor_gt_boxes[l], fpn_all_anchors[l])
            reg_targets_flatten.append(tf.reshape(anchor_gt_boxes_encoded, [-1, 4]))

        cls_logits_flatten = tf.concat(cls_logits_flatten, axis=0)
        reg_logits_flatten = tf.concat(reg_logits_flatten, axis=0)
        cls_labels_flatten = tf.concat(cls_labels_flatten, axis=0)
        reg_targets_flatten = tf.concat(reg_targets_flatten, axis=0)

        valid_indices = tf.reshape(tf.where(tf.not_equal(cls_labels_flatten, -1)), [-1])
        pos_indices = tf.reshape(tf.where(tf.greater(cls_labels_flatten, 0)), [-1])
        num_valid = tf.cast(tf.size(valid_indices), tf.int32)
        num_pos = tf.cast(tf.size(pos_indices), tf.int64)
        tf.summary.scalar('num_valid', num_valid)
        tf.summary.scalar('num_pos', num_pos)

        valid_cls_labels = tf.gather(cls_labels_flatten, valid_indices)
        # valid_cls_labels_onehot = tf.one_hot(valid_cls_labels - 1, depth=cfg.DATA.NUM_CATEGORY,
        #                                      axis=1)
        valid_cls_logits = tf.gather(cls_logits_flatten, valid_indices)

        placeholder = 0.
        # # cls_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        # #     labels=tf.cast(valid_cls_labels_onehot, tf.float32), logits=valid_cls_logits)
        # cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     labels=valid_cls_labels, logits=valid_cls_logits
        # )
        # cls_loss = tf.reduce_mean(cls_loss)
        cls_loss = sigmoid_focal_loss3(labels=valid_cls_labels, logits=valid_cls_logits)
        cls_loss = tf.where(tf.equal(num_valid, 0),
                            placeholder,
                            tf.truediv(cls_loss, tf.cast(num_pos, tf.float32)), name='cls_loss')

        pos_reg_targets_flatten = tf.gather(reg_targets_flatten, pos_indices)
        pos_reg_logits_flatten = tf.gather(reg_logits_flatten, pos_indices)
        box_loss = smooth_l1_loss(pos_reg_logits_flatten, pos_reg_targets_flatten, sigma=3.0)
        box_loss = tf.where(tf.equal(num_pos, 0),
                            placeholder,
                            tf.truediv(box_loss, tf.cast(num_pos, tf.float32)), name='box_loss')

        loss_dict = {'cls_loss': cls_loss,
                     'box_loss': box_loss}

        ###################################################################################
        if cfg.VISUALIZATION:
            final_boxes = []
            final_scores = []
            final_labels = []
            final_inds = []
            for level in range(len(cfg.RETINANET.ANCHOR_STRIDES)):
                final_boxes_, final_scores_, final_labels_, final_inds_ = \
                    output_predictions_per_level(tf.transpose(retinanet_outputs[level][0], [0, 2, 3, 1]),
                                                 tf.transpose(retinanet_outputs[level][1], [0, 2, 3, 1]),
                                                 fpn_all_anchors[level], height, width)
                final_boxes.append(final_boxes_)
                final_scores.append(final_scores_)
                final_labels.append(final_labels_)
                final_inds.append(final_inds_)
            final_boxes = tf.concat(final_boxes, axis=0)
            final_scores = tf.concat(final_scores, axis=0)
            final_labels = tf.concat(final_labels, axis=0)
            final_inds = tf.concat(final_inds, axis=0)

            with tf.device('/cpu:0'):
                with tf.name_scope('vis_preds'):
                    if data_format == 'channels_first':
                        images_show1 = tf.identity(tf.transpose(images, [0, 2, 3, 1]))  # NCHW --> NHWC
                    else:
                        images_show1 = tf.identity(images)
                    image_mean = tf.constant(cfg.PREPROC.PIXEL_MEAN, dtype=tf.float32)  # RGB
                    image_invstd = tf.constant(1.0 / np.asarray(cfg.PREPROC.PIXEL_STD), dtype=tf.float32)
                    images_show1 /= image_invstd
                    images_show1 = images_show1 + image_mean
                    images_show1 = tf.clip_by_value(images_show1 * cfg.PREPROC.PIXEL_SCALE, 0, 255)
                    images_show1 = tf.cast(images_show1, tf.uint8)

                    for b_id in range(cfg.TRAIN.BATCH_SIZE_PER_GPU):
                        cur_image = images_show1[b_id]
                        inds_this_image = tf.reshape(tf.where(tf.equal(final_inds, b_id)), [-1])
                        cur_final_boxes = tf.gather(final_boxes, inds_this_image)
                        cur_final_labels = tf.gather(final_labels, inds_this_image)
                        cur_final_scores = tf.gather(final_scores, inds_this_image)

                        pred_img = tf.py_func(draw_boxes_with_scores,
                                              inp=[cur_image, cur_final_boxes, cur_final_labels, cur_final_scores,
                                                   'pred_{}'.format(b_id)],
                                              Tout=[tf.uint8])
                        tf.summary.image('pred_img_{}'.format(b_id), pred_img)
        # ##################################################################################

        return loss_dict

    else:

        final_boxes = []
        final_scores = []
        final_labels = []
        final_inds = []
        for level in range(len(cfg.RETINANET.ANCHOR_STRIDES)):
            final_boxes_, final_scores_, final_labels_, final_inds_ = \
                output_predictions_per_level(retinanet_outputs[level][0],
                                             retinanet_outputs[level][1],
                                             fpn_all_anchors[level], height, width)
            final_boxes.append(final_boxes_)
            final_scores.append(final_scores_)
            final_labels.append(final_labels_)
            final_inds.append(final_inds_)
        final_boxes = tf.concat(final_boxes, axis=0)
        final_scores = tf.concat(final_scores, axis=0)
        final_labels = tf.concat(final_labels, axis=0)
        final_inds = tf.concat(final_inds, axis=0)

        return final_boxes, final_scores, final_labels, final_inds













