# -*- coding: utf-8 -*-
# @Time    : 2019/11/13 12:44
# @Author  : Zisha Zhong
# @Email   : zhongzisha@outlook.com
# @File    : fasterrcnn_model.py
# @Software: PyCharm


import math
import numpy as np
import tensorflow as tf
from config import cfg

from tensorflow.python.ops import gen_image_ops
# tf.image.non_max_suppression = gen_image_ops.non_max_suppression_v2

from viz import draw_on_img, draw_on_img_with_color, draw_heatmap, draw_boxes, draw_boxes_with_scores

TF_version = tuple(map(int, tf.__version__.split('.')[:2]))
from tensorflow.python.ops import gen_nccl_ops
from resnet_model import resnet_v1_c4_backbone, resnet_v1_c5, resnet_v1_fpn_backbone


def area(boxes):
    xmin, ymin, xmax, ymax = tf.split(boxes, num_or_size_splits=4, axis=1)
    return tf.squeeze((ymax - ymin) * (xmax - xmin), axis=[1])


def pairwise_intersection(boxlist1, boxlist2):
    # boxlist1: Nx4
    # boxlist2: Mx4
    xmin1, ymin1, xmax1, ymax1 = tf.split(boxlist1, num_or_size_splits=4, axis=1)
    xmin2, ymin2, xmax2, ymax2 = tf.split(boxlist2, num_or_size_splits=4, axis=1)
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
    return tf.where(tf.equal(intersections, 0.0), tf.zeros_like(intersections), tf.truediv(intersections, unions))


def get_all_anchors(height, width, stride, sizes, ratios):
    '''
    generate the anchors for feature map with certain stride
    :param height: image height
    :param width: image

    :param stride: stride for the feature map
    :param sizes: sqrt area of anchors
    :param ratios: h/w aspect ratios of anchors
    :return:
    '''
    anchors = []
    for sz in sizes:
        for ratio in ratios:
            w = np.sqrt(sz * sz / ratio)
            h = ratio * w
            anchors.append([-w, -h, w, h])
    cell_anchors = np.asarray(anchors) * 0.5
    feat_w = int(width // stride)
    feat_h = int(height // stride)
    shift_x = np.arange(feat_w) * stride # + stride // 2
    shift_y = np.arange(feat_h) * stride # + stride // 2
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    shifts = np.vstack((shift_x, shift_y, shift_x, shift_y)).transpose()
    K = shifts.shape[0]
    A = cell_anchors.shape[0]
    field_of_anchors = cell_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    field_of_anchors = field_of_anchors.reshape((feat_h, feat_w, A, 4))
    field_of_anchors = field_of_anchors.astype(np.float32)
    return field_of_anchors


def get_all_anchors_tf(height, width, stride, sizes, ratios):
    return None


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


def rpn_losses(label_logits, box_logits, anchor_gt_labels, anchor_gt_boxes_encoded):
    # label_logits: BSxfHxfWxNA
    # box_logits: BSxfHxfWxNAx4
    # anchor_labels: BSxfHxfWxNA
    # anchor_gt_boxes_encoded: BSxfHxfWxNAx4
    valid_mask = tf.stop_gradient(tf.not_equal(anchor_gt_labels, -1))
    pos_mask = tf.stop_gradient(tf.equal(anchor_gt_labels, 1))
    print('valid_mask', valid_mask)
    print('pos_mask', pos_mask)
    num_valid = tf.stop_gradient(tf.count_nonzero(valid_mask, dtype=tf.int32))
    num_pos = tf.identity(tf.count_nonzero(pos_mask, dtype=tf.int64))
    valid_anchor_labels = tf.boolean_mask(anchor_gt_labels, valid_mask)
    valid_label_logits = tf.boolean_mask(label_logits, valid_mask)

    tf.summary.scalar('num_valid', num_valid)
    tf.summary.scalar('num_pos', num_pos)

    with tf.name_scope('label_metrics'):
        valid_label_prob = tf.nn.sigmoid(valid_label_logits)  # BSxfHxfWxNA
        with tf.device('/cpu:0'):
            for th in [0.5, 0.2, 0.1]:
                valid_prediction = tf.cast(valid_label_prob > th, tf.int32)
                num_pos_prediction = tf.reduce_sum(valid_prediction)
                pos_prediction_corr = tf.count_nonzero(
                    tf.logical_and(valid_label_prob > th, tf.equal(valid_prediction, valid_anchor_labels))
                )
                placeholder = 0.5
                recall = tf.cast(tf.truediv(pos_prediction_corr, num_pos), tf.float32)
                recall = tf.where(tf.equal(num_pos, 0), placeholder, recall)
                precision = tf.cast(tf.truediv(pos_prediction_corr, tf.cast(num_pos_prediction, pos_prediction_corr.dtype)), tf.float32)
                precision = tf.where(tf.equal(num_pos_prediction, 0), placeholder, precision)
                tf.summary.scalar('recall_th{}'.format(th), recall)
                tf.summary.scalar('precision_th{}'.format(th), precision)

    placeholder = 0.
    cls_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.cast(valid_anchor_labels, tf.float32), logits=valid_label_logits)
    cls_loss = tf.reduce_sum(cls_loss) * (1. / (cfg.FRCNN.RPN.BATCH_PER_IM * cfg.TRAIN.BATCH_SIZE_PER_GPU))
    cls_loss = tf.where(tf.equal(num_valid, 0), placeholder, cls_loss, name='cls_loss')

    pos_anchor_boxes = tf.boolean_mask(tf.stop_gradient(anchor_gt_boxes_encoded), pos_mask)
    pos_box_logits = tf.boolean_mask(box_logits, pos_mask)
    print('pos_anchor_boxes', pos_anchor_boxes)
    print('pos_box_logits', pos_box_logits)
    # delta = 1.0 / 9
    # box_loss = tf.losses.huber_loss(
    #     labels=pos_anchor_boxes, predictions=pos_box_logits, delta=delta,
    #     reduction=tf.losses.Reduction.SUM) / delta
    box_loss = smooth_l1_loss(pos_box_logits, pos_anchor_boxes, sigma=3.0)
    box_loss = box_loss * (1. / (cfg.FRCNN.RPN.BATCH_PER_IM * cfg.TRAIN.BATCH_SIZE_PER_GPU))
    box_loss = tf.where(tf.equal(num_pos, 0), placeholder, box_loss, name='box_loss')

    return {'rpn_cls_loss': cls_loss,
            'rpn_box_loss': box_loss}


def rpn_losses_gather(label_logits, box_logits, anchor_gt_labels, anchor_gt_boxes_encoded):
    # label_logits: BSxfHxfWxNA
    # box_logits: BSxfHxfWxNAx4
    # anchor_labels: BSxfHxfWxNA
    # anchor_gt_boxes_encoded: BSxfHxfWxNAx4
    valid_indices = tf.stop_gradient(tf.where(tf.not_equal(anchor_gt_labels, -1)))
    pos_indices = tf.stop_gradient(tf.where(tf.equal(anchor_gt_labels, 1)))
    print('valid_indices', valid_indices)
    print('pos_indices', pos_indices)
    num_valid = tf.stop_gradient(tf.cast(tf.shape(valid_indices)[0], tf.int32))
    num_pos = tf.cast(tf.shape(pos_indices)[0], tf.int64)
    valid_anchor_labels = tf.gather_nd(anchor_gt_labels, valid_indices)
    valid_label_logits = tf.gather_nd(label_logits, valid_indices)

    tf.summary.scalar('num_valid', num_valid)
    tf.summary.scalar('num_pos', num_pos)

    with tf.name_scope('label_metrics'):
        valid_label_prob = tf.nn.sigmoid(valid_label_logits)  # BSxfHxfWxNA
        with tf.device('/cpu:0'):
            for th in [0.5, 0.2, 0.1]:
                valid_prediction = tf.cast(valid_label_prob > th, tf.int32)
                num_pos_prediction = tf.reduce_sum(valid_prediction)
                pos_prediction_corr = tf.count_nonzero(
                    tf.logical_and(valid_label_prob > th, tf.equal(valid_prediction, valid_anchor_labels))
                )
                placeholder = 0.5
                recall = tf.cast(tf.truediv(pos_prediction_corr, num_pos), tf.float32)
                recall = tf.where(tf.equal(num_pos, 0), placeholder, recall)
                precision = tf.cast(tf.truediv(pos_prediction_corr, tf.cast(num_pos_prediction, pos_prediction_corr.dtype)), tf.float32)
                precision = tf.where(tf.equal(num_pos_prediction, 0), placeholder, precision)
                tf.summary.scalar('recall_th{}'.format(th), recall)
                tf.summary.scalar('precision_th{}'.format(th), precision)

    placeholder = 0.
    cls_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.cast(valid_anchor_labels, tf.float32), logits=valid_label_logits)
    cls_loss = tf.reduce_sum(cls_loss) * (1. / (cfg.FRCNN.RPN.BATCH_PER_IM * cfg.TRAIN.BATCH_SIZE_PER_GPU))
    cls_loss = tf.where(tf.equal(num_valid, 0), placeholder, cls_loss, name='cls_loss')

    pos_anchor_boxes = tf.gather_nd(tf.stop_gradient(anchor_gt_boxes_encoded), pos_indices)
    pos_box_logits = tf.gather_nd(box_logits, pos_indices)
    print('pos_anchor_boxes', pos_anchor_boxes)
    print('pos_box_logits', pos_box_logits)
    # delta = 1.0 / 9
    # box_loss = tf.losses.huber_loss(
    #     labels=pos_anchor_boxes, predictions=pos_box_logits, delta=delta,
    #     reduction=tf.losses.Reduction.SUM) / delta
    box_loss = smooth_l1_loss(pos_box_logits, pos_anchor_boxes, sigma=3.0)
    box_loss = box_loss * (1. / (cfg.FRCNN.RPN.BATCH_PER_IM * cfg.TRAIN.BATCH_SIZE_PER_GPU))
    box_loss = tf.where(tf.equal(num_pos, 0), placeholder, box_loss, name='box_loss')

    return {'rpn_cls_loss': cls_loss,
            'rpn_box_loss': box_loss}


def clip_boxes(boxes, window):
    boxes = tf.maximum(boxes, 0.0)
    m = tf.tile(tf.reverse(window, [0]), [2])
    boxes = tf.minimum(boxes, tf.cast(m, tf.float32))
    return boxes


def generate_rpn_proposals(batch_boxes, batch_scores, img_height, img_width, pre_nms_topk, post_nms_topk=None):

    # boxes: BSxfHxfWxNAx4
    # scores: BSxfHxfWxNA
    if post_nms_topk is None:
        post_nms_topk = pre_nms_topk

    N = tf.shape(batch_boxes)[0]

    def condition(b_id, proposal_boxes, proposal_scores, proposal_boxes_inds):
        return tf.less(b_id, N)

    def body(b_id, proposal_boxes, proposal_scores, proposal_boxes_inds):
        boxes = tf.reshape(batch_boxes[b_id], [-1, 4])
        scores = tf.reshape(batch_scores[b_id], [-1])
        topk = tf.minimum(pre_nms_topk, tf.size(scores))
        topk_scores, topk_indices = tf.nn.top_k(scores, k=topk, sorted=False)
        topk_boxes = tf.gather(boxes, topk_indices)
        topk_boxes = clip_boxes(topk_boxes, (img_height, img_width))

        if cfg.FRCNN.RPN.MIN_SIZE > 0:
            topk_boxes_x1y1x2y2 = tf.reshape(topk_boxes, (-1, 2, 2))
            topk_boxes_x1y1, topk_boxes_x2y2 = tf.split(topk_boxes_x1y1x2y2, 2, axis=1)
            wbhb = tf.squeeze(topk_boxes_x2y2 - topk_boxes_x1y1, axis=1)
            valid = tf.reduce_all(wbhb > cfg.FRCNN.RPN.MIN_SIZE, axis=1)
            topk_valid_boxes = tf.boolean_mask(topk_boxes, valid)
            topk_valid_scores = tf.boolean_mask(topk_scores, valid)
        else:
            topk_valid_boxes = topk_boxes
            topk_valid_scores = topk_scores

        nms_indices = tf.image.non_max_suppression(topk_valid_boxes, topk_valid_scores,
                                                   max_output_size=post_nms_topk,
                                                   iou_threshold=cfg.FRCNN.RPN.PROPOSAL_NMS_THRESH)
        proposal_boxes_this_image = tf.gather(topk_valid_boxes, nms_indices)
        proposal_scores_this_image = tf.gather(topk_valid_scores, nms_indices)
        inds_this_image = b_id * tf.ones((tf.shape(proposal_boxes_this_image)[0],), dtype=tf.int32)

        proposal_boxes = tf.cond(tf.equal(b_id, 0),
                                 lambda: proposal_boxes_this_image,
                                 lambda: tf.concat([proposal_boxes, proposal_boxes_this_image], axis=0))
        proposal_scores = tf.cond(tf.equal(b_id, 0),
                                  lambda: proposal_scores_this_image,
                                  lambda: tf.concat([proposal_scores, proposal_scores_this_image], axis=0))
        proposal_boxes_inds = tf.cond(tf.equal(b_id, 0),
                                      lambda: inds_this_image,
                                      lambda: tf.concat([proposal_boxes_inds, inds_this_image], axis=0))

        return tf.add(b_id, 1), proposal_boxes, proposal_scores, proposal_boxes_inds

    b_id = tf.constant(0, dtype=tf.int32)
    proposal_boxes = tf.zeros([0, 4], dtype=tf.float32)
    proposal_scores = tf.zeros([0, ], dtype=tf.float32)
    proposal_boxes_inds = tf.zeros([0, ], dtype=tf.int32)
    index_results = (b_id, proposal_boxes, proposal_scores, proposal_boxes_inds)
    _, proposal_boxes, proposal_scores, proposal_boxes_inds = \
        tf.while_loop(condition, body, index_results,
                      shape_invariants=(b_id.get_shape(),
                                        tf.TensorShape([None, 4]),
                                        tf.TensorShape([None]),
                                        tf.TensorShape([None])))
    proposal_boxes = tf.reshape(proposal_boxes, [-1, 4])
    proposal_scores = tf.reshape(proposal_scores, [-1])
    proposal_boxes_inds = tf.reshape(proposal_boxes_inds, [-1])

    return tf.stop_gradient(proposal_boxes), \
           tf.stop_gradient(proposal_scores), \
           tf.stop_gradient(proposal_boxes_inds)


def generate_rpn_proposals_without_loop(batch_boxes, batch_scores, img_height, img_width, pre_nms_topk, post_nms_topk=None):

    # boxes: BSxfHxfWxNAx4
    # scores: BSxfHxfWxNA
    if post_nms_topk is None:
        post_nms_topk = pre_nms_topk
    N = tf.shape(batch_boxes)[0]
    boxes = tf.reshape(batch_boxes, [N, -1, 1, 4])
    scores = tf.reshape(batch_scores, [N, -1, 1])
    nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = \
        gen_image_ops.combined_non_max_suppression(boxes=boxes, scores=scores,
                                                   max_output_size_per_class=post_nms_topk,
                                                   max_total_size=post_nms_topk,
                                                   iou_threshold=cfg.FRCNN.RPN.PROPOSAL_NMS_THRESH,
                                                   score_threshold=0.01, clip_boxes=False)

    def condition(b_id, proposal_boxes, proposal_scores, proposal_boxes_inds):
        return tf.less(b_id, N)

    def body(b_id, proposal_boxes, proposal_scores, proposal_boxes_inds):
        boxes = tf.reshape(nmsed_boxes[b_id], [-1, 4])
        scores = tf.reshape(nmsed_scores[b_id], [-1])
        valid_inds = tf.reshape(tf.range(valid_detections[b_id]), [-1])
        proposal_boxes_this_image = tf.gather(boxes, valid_inds)
        proposal_scores_this_image = tf.gather(scores, valid_inds)

        inds_this_image = b_id * tf.ones((tf.shape(proposal_boxes_this_image)[0],), dtype=tf.int32)

        proposal_boxes = tf.cond(tf.equal(b_id, 0),
                                 lambda: proposal_boxes_this_image,
                                 lambda: tf.concat([proposal_boxes, proposal_boxes_this_image], axis=0))
        proposal_scores = tf.cond(tf.equal(b_id, 0),
                                  lambda: proposal_scores_this_image,
                                  lambda: tf.concat([proposal_scores, proposal_scores_this_image], axis=0))
        proposal_boxes_inds = tf.cond(tf.equal(b_id, 0),
                                      lambda: inds_this_image,
                                      lambda: tf.concat([proposal_boxes_inds, inds_this_image], axis=0))

        return tf.add(b_id, 1), proposal_boxes, proposal_scores, proposal_boxes_inds

    b_id = tf.constant(0, dtype=tf.int32)
    proposal_boxes = tf.zeros([0, 4], dtype=tf.float32)
    proposal_scores = tf.zeros([0, ], dtype=tf.float32)
    proposal_boxes_inds = tf.zeros([0, ], dtype=tf.int32)
    index_results = (b_id, proposal_boxes, proposal_scores, proposal_boxes_inds)
    _, proposal_boxes, proposal_scores, proposal_boxes_inds = \
        tf.while_loop(condition, body, index_results,
                      shape_invariants=(b_id.get_shape(),
                                        tf.TensorShape([None, 4]),
                                        tf.TensorShape([None]),
                                        tf.TensorShape([None])))
    proposal_boxes = tf.reshape(proposal_boxes, [-1, 4])
    proposal_scores = tf.reshape(proposal_scores, [-1])
    proposal_boxes_inds = tf.reshape(proposal_boxes_inds, [-1])

    return tf.stop_gradient(proposal_boxes), \
           tf.stop_gradient(proposal_scores), \
           tf.stop_gradient(proposal_boxes_inds)


def sample_fast_rcnn_targets(batch_boxes, batch_scores, batch_boxes_inds,
                             batch_gt_boxes=None,
                             batch_gt_labels=None,
                             batch_orig_gt_counts=None,
                             mode='train'):
    # boxes: nx4
    # boxes_inds: nx1
    # gt_boxes: BSxmx4
    # gt_labels: BSxmx4
    # orig_gt_counts: BSx1  number of gt boxes in each image

    N = tf.shape(batch_gt_boxes)[0]

    if cfg.MODE_FPN:
        fpn_post_nms_topk = cfg.FRCNN.RPN.TRAIN_PER_LEVEL_NMS_TOPK if mode == 'train' else \
                            cfg.FRCNN.RPN.TEST_PER_LEVEL_NMS_TOPK

    def condition(b_id, rois, roi_labels, roi_fg_targets, roi_inds):
        return tf.less(b_id, N)

    def body(b_id, rois, roi_labels, roi_fg_targets, roi_inds):
        inds_this_images = tf.reshape(tf.where(tf.equal(batch_boxes_inds, b_id)), [-1])
        boxes = tf.gather(batch_boxes, inds_this_images)
        scores = tf.gather(batch_scores, inds_this_images)

        if cfg.MODE_FPN:
            fpn_nms_topk = tf.minimum(tf.size(inds_this_images), fpn_post_nms_topk)
            topk_scores, topk_indices = tf.nn.top_k(scores, k=fpn_nms_topk, sorted=False)
            boxes = tf.gather(boxes, topk_indices)

        gt_boxes = batch_gt_boxes[b_id]
        gt_labels = batch_gt_labels[b_id]
        gt_count = batch_orig_gt_counts[b_id]

        valid_inds = tf.reshape(tf.range(gt_count), [-1])
        gt_boxes = tf.gather(gt_boxes, valid_inds)
        gt_labels = tf.gather(gt_labels, valid_inds)

        iou = pairwise_iou(boxes, gt_boxes)   # nxm
        boxes = tf.concat([boxes, gt_boxes], axis=0)   # (n+m)x4
        iou = tf.concat([iou, tf.eye(tf.shape(gt_boxes)[0])], axis=0)  # (n+m)xm

        def sample_bg_fg(iou):
            fg_mask = tf.cond(tf.shape(iou)[1] > 0,
                              lambda: tf.reduce_max(iou, axis=1) >= cfg.FRCNN.RCNN.FG_THRESH,
                              lambda: tf.zeros([tf.shape(iou)[0]], dtype=tf.bool))
            fg_inds = tf.reshape(tf.where(fg_mask), [-1])
            num_fg = tf.minimum(int(cfg.FRCNN.RCNN.BATCH_PER_IM * cfg.FRCNN.RCNN.FG_RATIO),
                                tf.size(fg_inds), name='num_fg')
            fg_inds = tf.random_shuffle(fg_inds)[:num_fg]

            bg_inds = tf.reshape(tf.where(tf.logical_not(fg_mask)), [-1])
            num_bg = tf.minimum(cfg.FRCNN.RPN.BATCH_PER_IM - num_fg,
                                tf.size(bg_inds), name='num_bg')
            bg_inds = tf.random_shuffle(bg_inds)[:num_bg]

            return fg_inds, bg_inds

        fg_inds, bg_inds = sample_bg_fg(iou)

        best_iou_ind = tf.cond(tf.shape(iou)[1] > 0,
                               lambda: tf.argmax(iou, axis=1),
                               lambda: tf.zeros((tf.shape(iou)[0]), dtype=tf.int64))   # (n+m)x1
        fg_inds_wrt_gt = tf.gather(best_iou_ind, fg_inds)   # numFG:  class indices of each proposal, in [0, M)

        all_indices = tf.concat([fg_inds, bg_inds], axis=0)  # (numFG + numBG)
        rois_this_image = tf.gather(boxes, all_indices)   # (numFG+numBG)x4

        roi_labels_this_image = tf.concat([tf.gather(gt_labels, fg_inds_wrt_gt),  # classes for each fg boxes
                                           tf.zeros_like(bg_inds, dtype=tf.int64)], axis=0)  # (numFG + numBG)

        fg_boxes_this_image = tf.gather(boxes, fg_inds)  # numFGx4
        fg_gt_boxes_this_image = tf.gather(gt_boxes, fg_inds_wrt_gt)  # numFGx4
        fg_box_targets_this_image = encode_bbox_target(fg_gt_boxes_this_image, fg_boxes_this_image)  # numFGx4

        roi_inds_this_image = b_id * tf.ones((tf.shape(rois_this_image)[0],), dtype=tf.int32)

        rois = tf.cond(tf.equal(b_id, 0),
                       lambda: rois_this_image,
                       lambda: tf.concat([rois, rois_this_image], axis=0))
        roi_labels = tf.cond(tf.equal(b_id, 0),
                             lambda: roi_labels_this_image,
                             lambda: tf.concat([roi_labels, roi_labels_this_image], axis=0))
        roi_fg_targets = tf.cond(tf.equal(b_id, 0),
                                 lambda: fg_box_targets_this_image,
                                 lambda: tf.concat([roi_fg_targets, fg_box_targets_this_image], axis=0))
        roi_inds = tf.cond(tf.equal(b_id, 0),
                           lambda: roi_inds_this_image,
                           lambda: tf.concat([roi_inds, roi_inds_this_image], axis=0))
        return tf.add(b_id, 1), rois, roi_labels, roi_fg_targets, roi_inds

    b_id = tf.constant(0, dtype=tf.int32)
    rois = tf.zeros([0, 4], dtype=tf.float32)
    roi_labels = tf.zeros([0, ], dtype=tf.int64)
    roi_fg_targets = tf.zeros([0, 4], dtype=tf.float32)
    roi_inds = tf.zeros([0, ], dtype=tf.int32)
    index_results = (b_id, rois, roi_labels, roi_fg_targets, roi_inds)
    _, rois, roi_labels, roi_fg_targets, roi_inds = \
        tf.while_loop(condition, body, index_results,
                      shape_invariants=(b_id.get_shape(),
                                        tf.TensorShape([None, 4]),
                                        tf.TensorShape([None]),
                                        tf.TensorShape([None, 4]),
                                        tf.TensorShape([None])))

    return tf.stop_gradient(rois),  \
           tf.stop_gradient(roi_labels), \
           tf.stop_gradient(roi_fg_targets), \
           tf.stop_gradient(roi_inds)


def roi_pooling(featuremap, boxes, box_inds, crop_size, data_format='channels_last'):
    if data_format == 'channels_first':
        shp2d = tf.shape(featuremap)[2:]
        featuremap = tf.transpose(featuremap, [0, 2, 3, 1])
    else:
        shp2d = tf.shape(featuremap)[1:3]
    feat_h, feat_w = tf.cast(shp2d[0], tf.float32), tf.cast(shp2d[1], tf.float32)
    xmin, ymin, xmax, ymax = tf.split(boxes, num_or_size_splits=4, axis=1)
    xmin /= feat_w
    ymin /= feat_h
    xmax /= feat_w
    ymax /= feat_h
    normalized_boxes = tf.concat([ymin, xmin, ymax, xmax], axis=1)
    normalized_boxes = tf.stop_gradient(normalized_boxes)

    box_features = tf.image.crop_and_resize(image=featuremap,
                                            boxes=normalized_boxes,
                                            box_ind=box_inds,
                                            crop_size=crop_size)  # nhwc
    box_features = tf.layers.average_pooling2d(inputs=box_features, pool_size=2,
                                               strides=2, padding='valid',
                                               data_format='channels_last')

    if data_format == 'channels_first':
        box_features = tf.transpose(box_features, [0, 3, 1, 2])  # nhwc --> nchw

    return box_features


def output_predictions(rois, roi_inds, box_deltas, box_scores, img_height=0, img_width=0):
    # box_deltas: nxCx4
    # box_scores: nxC

    N = tf.size(tf.unique(roi_inds)[0])
    num_classes = cfg.DATA.NUM_CATEGORY + 1

    def condition(b_id, final_boxes, final_scores, final_labels, final_inds):
        return tf.less(b_id, N)

    def body(b_id, final_boxes, final_scores, final_labels, final_inds):
        inds = tf.reshape(tf.where(tf.equal(roi_inds, b_id)), [-1])
        rois_this = tf.gather(rois, inds)
        box_deltas_this = tf.gather(box_deltas, inds)  # nxc4
        box_scores_this = tf.gather(box_scores, inds)  # nxc

        box_deltas_this = tf.transpose(tf.reshape(box_deltas_this, [-1, num_classes, 4]), [1, 0, 2])  #cxnx4
        box_scores_this = tf.transpose(tf.reshape(box_scores_this, [-1, num_classes]), [1, 0])  # cxn

        boxes_this_image = []
        scores_this_image = []
        labels_this_image = []

        for c in range(1, num_classes):
            boxes = tf.reshape(box_deltas_this[c], [-1, 4])  # nx4
            scores = tf.reshape(box_scores_this[c], [-1])  # n

            boxes = decode_bbox_target(boxes, rois_this)
            boxes = clip_boxes(boxes, (img_height, img_width))

            valid_inds = tf.reshape(tf.where(tf.greater_equal(scores, cfg.FRCNN.TEST.RESULT_SCORE_THRESH)), [-1])
            boxes = tf.gather(boxes, valid_inds)
            scores = tf.gather(scores, valid_inds)

            perclass_boxes = tf.reshape(boxes, [-1, 4])
            perclass_scores = tf.reshape(scores, [-1])

            keep = tf.image.non_max_suppression(perclass_boxes, perclass_scores,
                                                max_output_size=cfg.FRCNN.TEST.RESULTS_PER_IM,
                                                iou_threshold=cfg.FRCNN.TEST.NMS_THRESH)
            perclass_boxes = tf.gather(perclass_boxes, keep)
            perclass_scores = tf.gather(perclass_scores, keep)

            boxes_this_image.append(perclass_boxes)
            scores_this_image.append(perclass_scores)
            labels_this_image.append(c * tf.ones_like(perclass_scores, dtype=tf.int64))

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


def model(inputs, is_training=False, reuse=False, data_format='channels_last', mode='train'):

    loss_dict = {}

    if mode == 'train':
        images, gt_boxes, gt_labels, orig_gt_counts, anchor_gt_labels, anchor_gt_boxes, all_anchors = inputs

        ###################################################################################
        # visualization
        if cfg.FRCNN.VISUALIZATION:
            print('visualization gt')
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
                    gt_img = tf.py_func(draw_boxes, inp=[cur_image, cur_gt_boxes, cur_gt_labels],
                                        Tout=[tf.uint8])
                    tf.summary.image('gt_img_{}'.format(b_id), gt_img)
        ###################################################################################
    else:
        images = inputs

    if data_format == 'channels_first':
        images = tf.transpose(images, [0, 3, 1, 2])  # NHWC --> NCHW

    c4 = resnet_v1_c4_backbone(inputs=images, resnet_depth=50, is_training=is_training,
                               data_format=data_format, reuse=reuse)

    img_shape = tf.shape(images)
    feat_shape = c4.get_shape().as_list()
    if data_format == 'channels_first':
        height, width = img_shape[2], img_shape[3]
        feat_height, feat_width = feat_shape[2], feat_shape[3]
    else:
        height, width = img_shape[1], img_shape[2]
        feat_height, feat_width = feat_shape[1], feat_shape[2]

    num_anchors = int(len(cfg.FRCNN.ANCHOR.SIZES) * len(cfg.FRCNN.ANCHOR.RATIOS))

    if mode != 'train':
        all_anchors = tf.py_func(get_all_anchors, inp=[height, width, cfg.FRCNN.ANCHOR.STRIDE, cfg.FRCNN.ANCHOR.SIZES,
                                                       cfg.FRCNN.ANCHOR.RATIOS], Tout=tf.float32)
        all_anchors.set_shape([feat_height, feat_width, num_anchors, 4])

    if mode == 'train':
        ###################################################################################
        # visualization
        if cfg.FRCNN.VISUALIZATION:
            print('visualization anchors')
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

                all_anchors_for_show = tf.reshape(all_anchors, [-1, 4])

                for b_id in range(cfg.TRAIN.BATCH_SIZE_PER_GPU):
                    cur_image = images_show[b_id]
                    cur_anchor_labels = tf.reshape(anchor_gt_labels[b_id], [-1])
                    pos_inds = tf.reshape(tf.where(tf.equal(cur_anchor_labels, 1)), [-1])
                    neg_inds = tf.reshape(tf.where(tf.equal(cur_anchor_labels, 0)), [-1])

                    pos_anchor_boxes = tf.gather(all_anchors_for_show, pos_inds)
                    neg_anchor_boxes = tf.gather(all_anchors_for_show, neg_inds)

                    pos_anchor_labels = tf.gather(cur_anchor_labels, pos_inds)
                    neg_anchor_labels = tf.gather(cur_anchor_labels, neg_inds)

                    pos_img = tf.py_func(draw_boxes, inp=[cur_image, pos_anchor_boxes, pos_anchor_labels],
                                         Tout=[tf.uint8])
                    tf.summary.image('pos_anchor_img_{}'.format(b_id), pos_img)

                    neg_img = tf.py_func(draw_boxes, inp=[cur_image, neg_anchor_boxes, neg_anchor_labels],
                                         Tout=[tf.uint8])
                    tf.summary.image('neg_anchor_img_{}'.format(b_id), neg_img)
        ###################################################################################

        # anchor_gt_boxes_encoded = encode_bbox_target(anchor_gt_boxes, all_anchors)  # BS=1
        anchor_gt_boxes_encoded = encode_bbox_target_batch(anchor_gt_boxes, all_anchors)

    with tf.variable_scope('rpn'):
        rpn = tf.layers.conv2d(c4, filters=cfg.FRCNN.RPN.CHANNELS, kernel_size=3, padding='same',
                               name='conv', data_format=data_format,
                               kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                               activation=tf.nn.relu)
        rpn_cls_logits = tf.layers.conv2d(inputs=rpn, filters=num_anchors, kernel_size=1, padding='same',
                                          name='cls', data_format=data_format,
                                          kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        # BSxfHxfWx(NA*4)
        rpn_box_logits = tf.layers.conv2d(inputs=rpn, filters=num_anchors*4, kernel_size=1, padding='same',
                                          name='box', data_format=data_format,
                                          kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))

        if data_format == 'channels_first':
            rpn_cls_logits = tf.transpose(rpn_cls_logits, [0, 2, 3, 1])  # NCHW --> NHWC
            rpn_box_logits = tf.transpose(rpn_box_logits, [0, 2, 3, 1])  # NCHW --> NHWC

        shp = tf.shape(rpn_box_logits)  # BSxfHxfWx(NA*4)
        rpn_cls_logits = tf.reshape(rpn_cls_logits, [shp[0], shp[1], shp[2], num_anchors])
        rpn_box_logits = tf.reshape(rpn_box_logits, [shp[0], shp[1], shp[2], num_anchors, 4])

        if mode == 'train':
            rpn_loss_dict = rpn_losses_gather(rpn_cls_logits, rpn_box_logits, anchor_gt_labels, anchor_gt_boxes_encoded)
            loss_dict.update(rpn_loss_dict)

    pred_boxes_decoded = decode_bbox_target_batch(rpn_box_logits, all_anchors)  # BSxfHxfWxAx4

    proposal_boxes, proposal_scores, proposal_boxes_inds = generate_rpn_proposals(
        pred_boxes_decoded,
        rpn_cls_logits,
        height, width,
        cfg.FRCNN.RPN.TRAIN_PRE_NMS_TOPK if mode == 'train' else cfg.FRCNN.RPN.TEST_PRE_NMS_TOPK,
        cfg.FRCNN.RPN.TRAIN_POST_NMS_TOPK if mode == 'train' else cfg.FRCNN.RPN.TEST_POST_NMS_TOPK
    )

    if mode == 'train':
        # visualization
        if cfg.FRCNN.VISUALIZATION:
            print('visualization proposals')
            with tf.name_scope('vis_proposals'):
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
                    inds_this_image = tf.reshape(tf.where(tf.equal(proposal_boxes_inds, b_id)), [-1])
                    cur_roi_boxes = tf.gather(proposal_boxes, inds_this_image)
                    cur_roi_labels = tf.ones((tf.shape(cur_roi_boxes)[0],), dtype=tf.int64)
                    proposal_img = tf.py_func(draw_boxes, inp=[cur_image, cur_roi_boxes, cur_roi_labels],
                                         Tout=[tf.uint8])
                    tf.summary.image('proposal_img_{}'.format(b_id), proposal_img)
        ###################################################################################

        # Nx4
        rois, roi_labels, roi_fg_targets, roi_inds = \
            sample_fast_rcnn_targets(proposal_boxes, proposal_scores, proposal_boxes_inds,
                                     gt_boxes, gt_labels, orig_gt_counts)

        ###################################################################################
        # visualization
        if cfg.FRCNN.VISUALIZATION:
            print('visualization rois')
            with tf.name_scope('vis_rois'):
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
                    inds_this_image = tf.reshape(tf.where(tf.equal(roi_inds, b_id)), [-1])
                    cur_roi_boxes = tf.gather(rois, inds_this_image)
                    cur_roi_labels = tf.gather(roi_labels, inds_this_image)
                    pos_inds = tf.reshape(tf.where(tf.greater(cur_roi_labels, 0)), [-1])
                    neg_inds = tf.reshape(tf.where(tf.equal(cur_roi_labels, 0)), [-1])

                    pos_roi_boxes = tf.gather(cur_roi_boxes, pos_inds)
                    pos_roi_labels = tf.gather(cur_roi_labels, pos_inds)
                    pos_roi_img = tf.py_func(draw_boxes, inp=[cur_image, pos_roi_boxes, pos_roi_labels],
                                             Tout=[tf.uint8])
                    tf.summary.image('pos_roi_img_{}'.format(b_id), pos_roi_img)

                    neg_roi_boxes = tf.gather(cur_roi_boxes, neg_inds)
                    neg_roi_labels = tf.gather(cur_roi_labels, neg_inds)
                    neg_roi_img = tf.py_func(draw_boxes, inp=[cur_image, neg_roi_boxes, neg_roi_labels],
                                             Tout=[tf.uint8])
                    tf.summary.image('neg_roi_img_{}'.format(b_id), neg_roi_img)
        ##################################################################################

    else:
        # Nx4
        rois = proposal_boxes
        roi_inds = proposal_boxes_inds

    rois_on_featuremap = rois * (1.0 / cfg.FRCNN.ANCHOR.STRIDE)
    roi_resized = roi_pooling(c4, rois_on_featuremap, roi_inds, crop_size=[14, 14], data_format=data_format)

    # Nc77
    roi_features = resnet_v1_c5(roi_resized, resnet_depth=50, is_training=is_training,
                                data_format=data_format, reuse=reuse)

    num_classes = int(cfg.DATA.NUM_CATEGORY + 1)

    with tf.variable_scope('rcnn'):
        # Nc77 --> Nc
        pooled_feature = tf.layers.flatten(roi_features, data_format=data_format)  # no reduce_mean
        #
        head_feature = tf.layers.dense(inputs=pooled_feature, units=1024,
                                       kernel_initializer=tf.variance_scaling_initializer(), name='fc6',
                                       activation=tf.nn.relu)
        head_feature = tf.layers.dense(inputs=head_feature, units=1024,
                                       kernel_initializer=tf.variance_scaling_initializer(), name='fc7',
                                       activation=tf.nn.relu)
        # Nc --> NC
        rcnn_cls_logits = tf.layers.dense(inputs=head_feature, units=num_classes,
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.01), name='cls')
        # Nc --> N(C4)
        rcnn_box_logits = tf.layers.dense(inputs=head_feature, units=num_classes*4,
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.001), name='box')
        rcnn_box_logits = tf.reshape(rcnn_box_logits, [-1, num_classes, 4])

        box_regression_weights = tf.constant(cfg.FRCNN.RCNN.BBOX_REG_WEIGHTS, dtype=tf.float32)

        if mode == 'train':
            fg_inds = tf.reshape(tf.where(roi_labels > 0), [-1])
            fg_labels = tf.gather(roi_labels, fg_inds)
            num_fg = tf.size(fg_inds, out_type=tf.int64)
            tf.summary.scalar('rcnn_num_fg', num_fg)
            fg_box_logits = tf.gather(rcnn_box_logits, fg_inds)  # numFGxCx4
            indices = tf.stack([tf.range(num_fg), fg_labels], axis=1)
            print('indices', indices)
            fg_box_logits = tf.gather_nd(fg_box_logits, indices)
            fg_box_logits = tf.reshape(fg_box_logits, [-1, 4])  # numFGx4

            with tf.name_scope('label_metrics'), tf.device('/cpu:0'):
                empty_fg = tf.equal(num_fg, 0)
                prediction = tf.argmax(rcnn_cls_logits, axis=1)
                correct = tf.cast(tf.equal(prediction, roi_labels), tf.float32)
                accuracy = tf.reduce_mean(correct)
                fg_label_pred = tf.argmax(tf.gather(rcnn_cls_logits, fg_inds), axis=1)
                num_zero = tf.reduce_sum(tf.cast(tf.equal(fg_label_pred, 0), tf.int64))
                false_negative = tf.where(empty_fg, 0., tf.cast(tf.truediv(num_zero, num_fg), tf.float32))
                fg_accuracy = tf.where(empty_fg, 0., tf.reduce_mean(tf.gather(correct, fg_inds)))

                tf.summary.scalar('accuracy', accuracy)
                tf.summary.scalar('false_negative', false_negative)
                tf.summary.scalar('fg_accuracy', fg_accuracy)

            rcnn_cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=roi_labels, logits=rcnn_cls_logits)
            rcnn_cls_loss = tf.reduce_mean(rcnn_cls_loss)

            # rcnn_box_loss = tf.reduce_sum(tf.abs(roi_fg_targets * box_regression_weights - fg_box_logits))
            rcnn_box_loss = smooth_l1_loss(fg_box_logits, roi_fg_targets * box_regression_weights, sigma=3.0)
            rcnn_box_loss = tf.truediv(rcnn_box_loss, tf.cast(tf.shape(roi_labels)[0], tf.float32))

            rcnn_loss_dict = {'rcnn_cls_loss': rcnn_cls_loss, 'rcnn_box_loss': rcnn_box_loss}

            loss_dict.update(rcnn_loss_dict)

        ###################################################################################
        if mode == 'train' and cfg.FRCNN.VISUALIZATION:
            rcnn_cls_scores = tf.nn.softmax(rcnn_cls_logits)
            final_boxes, final_scores, final_labels, final_inds = \
                output_predictions(rois, roi_inds,
                                   rcnn_box_logits / box_regression_weights,
                                   rcnn_cls_scores, height, width)
            print('visualization preds')
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
                                          inp=[cur_image, cur_final_boxes, cur_final_labels, cur_final_scores],
                                          Tout=[tf.uint8])
                    tf.summary.image('pred_img_{}'.format(b_id), pred_img)
        # ##################################################################################

    if mode == 'train':
        return loss_dict
    else:
        rcnn_cls_scores = tf.nn.softmax(rcnn_cls_logits)
        final_boxes, final_scores, final_labels, final_inds = \
            output_predictions(rois, roi_inds,
                               rcnn_box_logits / box_regression_weights,
                               rcnn_cls_scores, height, width)
        return final_boxes, final_scores, final_labels, final_inds


def get_inputs(mode=0):

    if mode == 0:
        inputs = []
        inputs.append(tf.placeholder(tf.float32, shape=[None, None, None, 3]))          # images
        inputs.append(tf.placeholder(tf.float32, shape=[None, None, 4]))                # gt_boxes
        inputs.append(tf.placeholder(tf.int64, shape=[None, None]))                     # gt_labels
        inputs.append(tf.placeholder(tf.int32, shape=[None, ]))                         # orig_gt_counts
        inputs.append(tf.placeholder(tf.int32, shape=[None, None, None, None]))         # anchor_gt_labels
        inputs.append(tf.placeholder(tf.float32, shape=[None, None, None, None, 4]))    # anchor_gt_boxes

        if cfg.MODE_FPN:
            pass

        return inputs
    elif mode == 1:
        inputs_names = ['images', 'gt_boxes', 'gt_labels', 'orig_gt_counts',
                        'anchor_labels', 'anchor_boxes']
        if cfg.MODE_FPN:
            pass

        return inputs_names


def assign_boxes_to_level(boxes, box_inds):
    # rois: nx4
    xmin, ymin, xmax, ymax = tf.split(boxes, num_or_size_splits=4, axis=1)
    areas = tf.squeeze((ymax - ymin) * (xmax - xmin), axis=1)
    sqrtareas = tf.sqrt(areas)
    level = tf.cast(tf.math.floor(4 + tf.math.log(sqrtareas * (1. / 224) + 1e-6) * (1.0 / np.log(2))), tf.int32)
    level_ids = [
        tf.where(tf.less_equal(level, 2)),
        tf.where(tf.equal(level, 3)),
        tf.where(tf.equal(level, 4)),
        tf.where(tf.greater_equal(level, 5))
    ]
    level_ids = [tf.reshape(x, [-1]) for i, x in enumerate(level_ids)]
    num_in_levels = [tf.size(x) for i, x in enumerate(level_ids)]
    level_boxes = [tf.gather(boxes, ids) for ids in level_ids]
    level_box_inds = [tf.gather(box_inds, ids) for ids in level_ids]
    return level_ids, level_boxes, level_box_inds


def model_fpn(inputs, is_training=False, reuse=False, data_format='channels_last', mode='train'):

    loss_dict = {}

    if mode == 'train':
        images, gt_boxes, gt_labels, orig_gt_counts, fpn_all_anchors, \
        fpn_anchor_gt_labels, fpn_anchor_gt_boxes = inputs
        # fpn_all_anchors is a list
        # fpn_anchor_gt_labels is a list
        # fpn_anchor_gt_boxes is a list

        ###################################################################################
        # visualization
        if cfg.FRCNN.VISUALIZATION:
            print('visualization gt')
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

    features = resnet_v1_fpn_backbone(inputs=images, resnet_depth=50, is_training=is_training,
                                      data_format=data_format, reuse=reuse)

    img_shape = tf.shape(images)
    if data_format == 'channels_first':
        height, width = img_shape[2], img_shape[3]
    else:
        height, width = img_shape[1], img_shape[2]

    if mode != 'train':
        fpn_all_anchors = []
        for idx, (stride, size) in enumerate(zip(cfg.FRCNN.FPN.ANCHOR_STRIDES, cfg.FRCNN.ANCHOR.SIZES)):
            feature_shape_this_level = features[idx].get_shape().as_list()
            if data_format == 'channels_first':
                feat_height, feat_width = feature_shape_this_level[2], feature_shape_this_level[3]
            else:
                feat_height, feat_width = feature_shape_this_level[1], feature_shape_this_level[2]

            sizes = (size, )
            num_anchors = int(len(sizes) * len(cfg.FRCNN.ANCHOR.RATIOS))

            anchors_this_level = tf.py_func(get_all_anchors, inp=[height, width, stride, sizes,
                                                                  cfg.FRCNN.ANCHOR.RATIOS], Tout=tf.float32)
            anchors_this_level.set_shape([feat_height, feat_width, num_anchors, 4])

            fpn_all_anchors.append(anchors_this_level)

    if mode == 'train':
        ###################################################################################
        # visualization
        if cfg.FRCNN.VISUALIZATION:
            print('visualization anchors')
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
                    with tf.name_scope('Level-{}'.format(2 + idx)):
                        all_anchors_for_show = tf.reshape(fpn_all_anchors[idx], [-1, 4])

                        for b_id in range(cfg.TRAIN.BATCH_SIZE_PER_GPU):
                            cur_image = images_show[b_id]
                            cur_anchor_labels = tf.reshape(fpn_anchor_gt_labels[idx][b_id], [-1])
                            pos_inds = tf.reshape(tf.where(tf.equal(cur_anchor_labels, 1)), [-1])
                            neg_inds = tf.reshape(tf.where(tf.equal(cur_anchor_labels, 0)), [-1])

                            pos_anchor_boxes = tf.gather(all_anchors_for_show, pos_inds)
                            neg_anchor_boxes = tf.gather(all_anchors_for_show, neg_inds)

                            pos_anchor_labels = tf.gather(cur_anchor_labels, pos_inds)
                            neg_anchor_labels = tf.gather(cur_anchor_labels, neg_inds)

                            pos_img = tf.py_func(draw_boxes, inp=[cur_image, pos_anchor_boxes, pos_anchor_labels,
                                                                  'pos_anchor_{}_{}'.format(idx+2, b_id)],
                                                 Tout=[tf.uint8])
                            tf.summary.image('pos_anchor_img_{}'.format(b_id), pos_img)

                            neg_img = tf.py_func(draw_boxes, inp=[cur_image, neg_anchor_boxes, neg_anchor_labels,
                                                                  'neg_anchor_{}_{}'.format(idx+2, b_id)],
                                                 Tout=[tf.uint8])
                            tf.summary.image('neg_anchor_img_{}'.format(b_id), neg_img)
        ###################################################################################

    rpn_cls_losses = []
    rpn_box_losses = []
    proposal_boxes = []
    proposal_scores = []
    proposal_boxes_inds = []
    for idx, (stride, size) in enumerate(zip(cfg.FRCNN.FPN.ANCHOR_STRIDES, cfg.FRCNN.ANCHOR.SIZES)):
        feature = features[idx]
        num_anchors = int(len((size,)) * len(cfg.FRCNN.ANCHOR.RATIOS))
        with tf.variable_scope('rpn', reuse=(idx > 0)):
            rpn = tf.layers.conv2d(feature, filters=cfg.FRCNN.FPN.NUM_CHANNEL, kernel_size=3, padding='same',
                                   name='conv', data_format=data_format,
                                   kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                   activation=tf.nn.relu)
            rpn_cls_logits = tf.layers.conv2d(inputs=rpn, filters=num_anchors, kernel_size=1, padding='same',
                                              name='cls', data_format=data_format,
                                              kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            # BSxfHxfWx(NA*4)
            rpn_box_logits = tf.layers.conv2d(inputs=rpn, filters=num_anchors*4, kernel_size=1, padding='same',
                                              name='box', data_format=data_format,
                                              kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))

            if data_format == 'channels_first':
                rpn_cls_logits = tf.transpose(rpn_cls_logits, [0, 2, 3, 1])  # NCHW --> NHWC
                rpn_box_logits = tf.transpose(rpn_box_logits, [0, 2, 3, 1])  # NCHW --> NHWC

            shp = tf.shape(rpn_box_logits)  # BSxfHxfWx(NA*4)
            rpn_cls_logits = tf.reshape(rpn_cls_logits, [shp[0], shp[1], shp[2], num_anchors])
            rpn_box_logits = tf.reshape(rpn_box_logits, [shp[0], shp[1], shp[2], num_anchors, 4])

            pred_boxes_decoded = decode_bbox_target_batch(rpn_box_logits, fpn_all_anchors[idx])  # BSxfHxfWxAx4
            proposal_boxes_this_level, proposal_scores_this_level, proposal_boxes_inds_this_level = \
                generate_rpn_proposals(pred_boxes_decoded,
                                       rpn_cls_logits,
                                       height, width,
                                       cfg.FRCNN.RPN.TRAIN_PER_LEVEL_NMS_TOPK if mode == 'train'
                                       else cfg.FRCNN.RPN.TEST_PER_LEVEL_NMS_TOPK
                                       )
            proposal_boxes.append(proposal_boxes_this_level)
            proposal_scores.append(proposal_scores_this_level)
            proposal_boxes_inds.append(proposal_boxes_inds_this_level)

        if mode == 'train':
            anchor_gt_boxes_encoded = encode_bbox_target_batch(fpn_anchor_gt_boxes[idx], fpn_all_anchors[idx])
            rpn_loss_dict_this_level = rpn_losses(rpn_cls_logits, rpn_box_logits,
                                                  fpn_anchor_gt_labels[idx], anchor_gt_boxes_encoded)
            rpn_cls_losses.append(rpn_loss_dict_this_level['rpn_cls_loss'])
            rpn_box_losses.append(rpn_loss_dict_this_level['rpn_box_loss'])

    if mode == 'train':
        rpn_loss_dict = {'rpn_cls_loss': tf.add_n(rpn_cls_losses),
                         'rpn_box_loss': tf.add_n(rpn_box_losses)}
        loss_dict.update(rpn_loss_dict)

    proposal_boxes = tf.concat(proposal_boxes, axis=0)
    proposal_scores = tf.concat(proposal_scores, axis=0)
    proposal_boxes_inds = tf.concat(proposal_boxes_inds, axis=0)

    if mode == 'train':
        # visualization
        if cfg.FRCNN.VISUALIZATION:
            print('visualization proposals')
            with tf.name_scope('vis_proposals'):
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
                    inds_this_image = tf.reshape(tf.where(tf.equal(proposal_boxes_inds, b_id)), [-1])
                    cur_roi_boxes = tf.gather(proposal_boxes, inds_this_image)
                    cur_roi_labels = tf.ones((tf.shape(cur_roi_boxes)[0],), dtype=tf.int64)
                    proposal_img = tf.py_func(draw_boxes, inp=[cur_image, cur_roi_boxes, cur_roi_labels,
                                                               'proposal_{}'.format(b_id)],
                                              Tout=[tf.uint8])
                    tf.summary.image('proposal_img_{}'.format(b_id), proposal_img)
        ###################################################################################

        # Nx4
        rois, roi_labels, roi_fg_targets, roi_inds = \
            sample_fast_rcnn_targets(proposal_boxes, proposal_scores, proposal_boxes_inds,
                                     gt_boxes, gt_labels, orig_gt_counts)

        ###################################################################################
        # visualization
        if cfg.FRCNN.VISUALIZATION:
            print('visualization rois')
            with tf.name_scope('vis_rois'):
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
                    inds_this_image = tf.reshape(tf.where(tf.equal(roi_inds, b_id)), [-1])
                    cur_roi_boxes = tf.gather(rois, inds_this_image)
                    cur_roi_labels = tf.gather(roi_labels, inds_this_image)
                    pos_inds = tf.reshape(tf.where(tf.greater(cur_roi_labels, 0)), [-1])
                    neg_inds = tf.reshape(tf.where(tf.equal(cur_roi_labels, 0)), [-1])

                    pos_roi_boxes = tf.gather(cur_roi_boxes, pos_inds)
                    pos_roi_labels = tf.gather(cur_roi_labels, pos_inds)
                    pos_roi_img = tf.py_func(draw_boxes, inp=[cur_image, pos_roi_boxes, pos_roi_labels,
                                                              'pos_roi_{}'.format(b_id)],
                                             Tout=[tf.uint8])
                    tf.summary.image('pos_roi_img_{}'.format(b_id), pos_roi_img)

                    neg_roi_boxes = tf.gather(cur_roi_boxes, neg_inds)
                    neg_roi_labels = tf.gather(cur_roi_labels, neg_inds)
                    neg_roi_img = tf.py_func(draw_boxes, inp=[cur_image, neg_roi_boxes, neg_roi_labels,
                                                              'neg_roi_{}'.format(b_id)],
                                             Tout=[tf.uint8])
                    tf.summary.image('neg_roi_img_{}'.format(b_id), neg_roi_img)
        ##################################################################################

    else:
        # Nx4
        rois = proposal_boxes
        roi_inds = proposal_boxes_inds

    level_ids, level_rois, level_roi_inds = assign_boxes_to_level(rois, tf.reshape(roi_inds, [-1, 1]))
    roi_features = []
    for idx, (rois_level, roi_inds_level) in enumerate(zip(level_rois, level_roi_inds)):
        feature = features[idx]
        print(idx, rois_level, feature)
        rois_on_featuremap = rois_level * (1.0 / cfg.FRCNN.FPN.ANCHOR_STRIDES[idx])
        roi_features_level = roi_pooling(feature, rois_on_featuremap, tf.reshape(roi_inds_level, [-1]),
                                         crop_size=[14, 14], data_format=data_format)
        roi_features.append(roi_features_level)
    roi_features = tf.concat(roi_features, axis=0)
    level_id_perm = tf.concat(level_ids, axis=0)
    level_roi_inds = tf.concat(level_roi_inds, axis=0)
    level_id_invert_perm = tf.invert_permutation(level_id_perm)
    roi_features = tf.gather(roi_features, level_id_invert_perm)
    roi_inds = tf.reshape(tf.gather(level_roi_inds, level_id_invert_perm), [-1])

    num_classes = int(cfg.DATA.NUM_CATEGORY + 1)

    with tf.variable_scope('rcnn'):
        # Nc77 --> Nc
        pooled_feature = tf.layers.flatten(roi_features, data_format=data_format)
        fc6 = tf.layers.dense(inputs=pooled_feature, units=1024,
                              kernel_initializer=tf.variance_scaling_initializer(), name='fc6',
                              activation=tf.nn.relu)
        head_feature = tf.layers.dense(inputs=fc6, units=1024,
                                       kernel_initializer=tf.variance_scaling_initializer(), name='fc7',
                                       activation=tf.nn.relu)
        # Nc --> NC
        rcnn_cls_logits = tf.layers.dense(inputs=head_feature, units=num_classes,
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.01), name='cls')
        # Nc --> N(C4)
        rcnn_box_logits = tf.layers.dense(inputs=head_feature, units=num_classes*4,
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.001), name='box')
        rcnn_box_logits = tf.reshape(rcnn_box_logits, [-1, num_classes, 4])

        box_regression_weights = tf.constant(cfg.FRCNN.RCNN.BBOX_REG_WEIGHTS, dtype=tf.float32)

        if mode == 'train':
            fg_inds = tf.reshape(tf.where(roi_labels > 0), [-1])
            fg_labels = tf.gather(roi_labels, fg_inds)
            num_fg = tf.size(fg_inds, out_type=tf.int64)
            tf.summary.scalar('rcnn_num_fg', num_fg)
            fg_box_logits = tf.gather(rcnn_box_logits, fg_inds)  # numFGxCx4
            indices = tf.stack([tf.range(num_fg), fg_labels], axis=1)
            print('indices', indices)
            fg_box_logits = tf.gather_nd(fg_box_logits, indices)
            fg_box_logits = tf.reshape(fg_box_logits, [-1, 4])  # numFGx4

            with tf.name_scope('label_metrics'), tf.device('/cpu:0'):
                empty_fg = tf.equal(num_fg, 0)
                prediction = tf.argmax(rcnn_cls_logits, axis=1)
                correct = tf.cast(tf.equal(prediction, roi_labels), tf.float32)
                accuracy = tf.reduce_mean(correct)
                fg_label_pred = tf.argmax(tf.gather(rcnn_cls_logits, fg_inds), axis=1)
                num_zero = tf.reduce_sum(tf.cast(tf.equal(fg_label_pred, 0), tf.int64))
                false_negative = tf.where(empty_fg, 0., tf.cast(tf.truediv(num_zero, num_fg), tf.float32))
                fg_accuracy = tf.where(empty_fg, 0., tf.reduce_mean(tf.gather(correct, fg_inds)))

                tf.summary.scalar('accuracy', accuracy)
                tf.summary.scalar('false_negative', false_negative)
                tf.summary.scalar('fg_accuracy', fg_accuracy)

            rcnn_cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=roi_labels, logits=rcnn_cls_logits)
            rcnn_cls_loss = tf.reduce_mean(rcnn_cls_loss)

            # rcnn_box_loss = tf.reduce_sum(tf.abs(roi_fg_targets * box_regression_weights - fg_box_logits))
            rcnn_box_loss = smooth_l1_loss(fg_box_logits, roi_fg_targets * box_regression_weights, sigma=3.0)
            rcnn_box_loss = tf.truediv(rcnn_box_loss, tf.cast(tf.shape(roi_labels)[0], tf.float32))

            rcnn_loss_dict = {'rcnn_cls_loss': rcnn_cls_loss,
                              'rcnn_box_loss': rcnn_box_loss}

            loss_dict.update(rcnn_loss_dict)

        if mode == 'train' and cfg.FRCNN.VISUALIZATION:
        ###################################################################################
            rcnn_cls_scores = tf.nn.softmax(rcnn_cls_logits)
            final_boxes, final_scores, final_labels, final_inds = \
                output_predictions(rois, roi_inds,
                                   rcnn_box_logits / box_regression_weights,
                                   rcnn_cls_scores, height, width)

            print('visualization preds')
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

    if mode == 'train':
        return loss_dict
    else:
        rcnn_cls_scores = tf.nn.softmax(rcnn_cls_logits)
        final_boxes, final_scores, final_labels, final_inds = \
            output_predictions(rois, roi_inds,
                               rcnn_box_logits / box_regression_weights,
                               rcnn_cls_scores, height, width)
        return final_boxes, final_scores, final_labels, final_inds


if __name__ == '__main__':

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    data = dict(np.load('data_sample.npz'))
    images = tf.constant(data['images'])
    gt_boxes = tf.constant(data['gt_boxes'])
    gt_labels = tf.constant(data['gt_labels'])
    orig_gt_counts = tf.constant(data['orig_gt_counts'])
    anchor_gt_labels = tf.constant(data['anchor_gt_labels'])
    anchor_gt_boxes = tf.constant(data['anchor_gt_boxes'])

    inputs = (images, gt_boxes, gt_labels, orig_gt_counts, anchor_gt_labels, anchor_gt_boxes)

    sess_config = tf.ConfigProto()
    sess_config.allow_soft_placement = True
    sess_config.log_device_placement = False
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    loss_dict = model(inputs, is_training=False, reuse=False, data_format=cfg.BACKBONE.DATA_FORMAT, mode='train')

    init_op = tf.group([tf.global_variables_initializer(), tf.local_variables_initializer()])
    sess.run(init_op)

    total_loss = tf.add_n([v for k, v in loss_dict.items()])
    rpn_cls_loss, rpn_box_loss, rcnn_cls_loss, rcnn_box_loss = \
        sess.run([v for k, v in loss_dict.items()])
    print(rpn_cls_loss, rpn_box_loss, rcnn_cls_loss, rcnn_box_loss)










