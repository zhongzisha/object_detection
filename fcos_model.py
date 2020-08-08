# -*- coding: utf-8 -*-
# @Time    : 2019/11/7 16:35
# @Author  : Zisha Zhong
# @Email   : zhongzisha@outlook.com
# @File    : fcos_utils.py
# @Software: PyCharm
import math
import numpy as np
import tensorflow as tf
from config import cfg

from viz import draw_on_img, draw_on_img_with_color, draw_heatmap, draw_boxes

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

INF = 100000000


def compute_locations(features, data_format='channels_first'):
    locations = []
    for level, feature in enumerate(features):
        if data_format == 'channels_first':
            shp2d = tf.cast(tf.shape(feature)[2:], tf.float32)
        else:
            shp2d = tf.cast(tf.shape(feature)[1:3], tf.float32)
        h, w = shp2d[0], shp2d[1]
        stride = cfg.FCOS.FPN_STRIDES[level]
        shifts_x = tf.range(start=0, limit=w * stride, delta=stride, dtype=tf.float32)
        shifts_y = tf.range(start=0, limit=h * stride, delta=stride, dtype=tf.float32)
        shift_y, shift_x = tf.meshgrid(shifts_y, shifts_x, indexing='ij')
        shift_y = tf.reshape(shift_y, [-1])
        shift_x = tf.reshape(shift_x, [-1])
        locations_per_level = tf.stack([shift_x, shift_y], axis=1) + stride // 2
        locations.append(tf.stop_gradient(locations_per_level))
    return locations


def prepare_targets(images, points, gt_boxes, gt_labels, orig_gt_counts, data_format='channels_first'):
    object_sizes_of_interest = [
        [-1, 64],
        [64, 128],
        [128, 256],
        [256, 512],
        [512, INF]
    ]

    expanded_object_sizes_of_interest = []
    num_points_per_level = []
    for l, points_per_level in enumerate(points):
        object_sizes_of_interest_per_level = \
            tf.ones((tf.shape(points_per_level)[0], 2), dtype=tf.float32) * object_sizes_of_interest[l]
        expanded_object_sizes_of_interest.append(object_sizes_of_interest_per_level)
        num_points_per_level.append(tf.shape(points_per_level)[0])

    expanded_object_sizes_of_interest = tf.concat(expanded_object_sizes_of_interest, axis=0)
    points_all_level = tf.concat(points, axis=0)
    labels, reg_targets = compute_targets_for_locations(points_all_level, gt_boxes, gt_labels, orig_gt_counts,
                                                        expanded_object_sizes_of_interest)

    for i in range(len(labels)):
        labels[i] = tf.split(labels[i], num_points_per_level, axis=0)
        reg_targets[i] = tf.split(reg_targets[i], num_points_per_level, axis=0)

    #########################################################################
    # visualization
    if cfg.FCOS.VISUALIZATION:

        print('visualization ')

        if data_format == 'channels_first':
            images_show = tf.identity(tf.transpose(images, [0, 2, 3, 1]))  # NCHW --> NHWC
        else:
            images_show = tf.identity(images)
        image_mean = tf.constant(cfg.PREPROC.PIXEL_MEAN, dtype=tf.float32)  # BGR
        image_invstd = tf.constant(1.0 / np.asarray(cfg.PREPROC.PIXEL_STD), dtype=tf.float32)
        images_show /= image_invstd
        images_show = images_show + image_mean
        images_show = tf.clip_by_value(images_show * cfg.PREPROC.PIXEL_SCALE, 0, 255)
        images_show = tf.cast(images_show, tf.uint8)

        with tf.device('/cpu:0'):
            with tf.name_scope('vis'):
                for im_i in range(cfg.TRAIN.BATCH_SIZE_PER_GPU):
                    with tf.name_scope('image_{}'.format(im_i)):
                        cur_image = tf.expand_dims(images_show[im_i], axis=0)  # 1HWC
                        img_shp2d = tf.shape(cur_image)[1:3]
                        img_h, img_w = img_shp2d[0], img_shp2d[1]
                        for level in range(len(cfg.FCOS.FPN_STRIDES)):
                            with tf.name_scope('P_{}'.format(level + 3)):
                                locations_this_level = points[level]  # n
                                labels_this_level = labels[im_i][level]  # n
                                reg_targets_this_level = reg_targets[im_i][level]  # nx4

                                pos_indices = tf.reshape(tf.where(tf.greater(labels_this_level, 0)), [-1])
                                locations_this_level = tf.gather(locations_this_level, pos_indices)
                                labels_this_level = tf.gather(labels_this_level, pos_indices)
                                reg_targets_this_level = tf.gather(reg_targets_this_level, pos_indices)

                                # 画框的标签
                                target_im_this_level = draw_on_img_with_color(img_h, img_w, locations_this_level,
                                                                              labels_this_level)
                                target_im_this_level = tf.reshape(target_im_this_level, [1, img_w, img_h, 3])
                                target_im_this_level = tf.transpose(target_im_this_level, [0, 2, 1, 3])
                                if TF_version <= (1, 14):
                                    tf.summary.image('cls_gt', target_im_this_level)
                                else:
                                    tf.compat.v1.summary.image('cls_gt', target_im_this_level)

                                # 画框的回归GT
                                left, top, right, bottom = tf.split(reg_targets_this_level, num_or_size_splits=4,
                                                                    axis=1)
                                temp = {'left': left, 'right': right, 'top': top, 'bottom': bottom}
                                for key in temp.keys():
                                    target_im_this_level = draw_on_img(img_h, img_w, locations_this_level, temp[key])
                                    target_im_this_level = tf.reshape(target_im_this_level, [1, img_w, img_h, 1])
                                    target_im_this_level = tf.transpose(target_im_this_level, [0, 2, 1, 3])
                                    if TF_version <= (1, 14):
                                        tf.summary.image('reg_gt_{}'.format(key), target_im_this_level)
                                    else:
                                        tf.compat.v1.summary.image('reg_gt_{}'.format(key), target_im_this_level)

    labels_level_first = []
    reg_targets_level_first = []
    for level in range(len(num_points_per_level)):
        labels_this_level = []
        for label_per_im in labels:
            labels_this_level.append(label_per_im[level])
        labels_this_level = tf.concat(labels_this_level, axis=0)

        labels_this_level = tf.stop_gradient(labels_this_level)
        labels_level_first.append(labels_this_level)

        reg_targets_this_level = []
        for reg_targets_per_im in reg_targets:
            reg_targets_this_level.append(reg_targets_per_im[level])
        if cfg.FCOS.NORM_REG_TARGETS:
            reg_targets_this_level = tf.concat(reg_targets_this_level, axis=0) / cfg.FCOS.FPN_STRIDES[level]

        reg_targets_this_level = tf.stop_gradient(reg_targets_this_level)
        reg_targets_level_first.append(reg_targets_this_level)

    return labels_level_first, reg_targets_level_first


def compute_targets_for_locations(locations, gt_boxes, gt_labels, orig_gt_counts,
                                  object_sizes_of_interest):
    labels = []
    reg_targets = []
    xs, ys = tf.split(locations, num_or_size_splits=[1, 1], axis=1)
    for i in range(cfg.TRAIN.BATCH_SIZE_PER_GPU):
        bboxes = gt_boxes[i]  # mx4
        label = gt_labels[i]  # m
        gt_count = orig_gt_counts[i]

        valid_inds = tf.reshape(tf.range(gt_count), [-1])
        bboxes = tf.gather(bboxes, valid_inds)
        label = tf.gather(label, valid_inds)

        area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])  # m

        left = xs - bboxes[:, 0]  # nxm
        top = ys - bboxes[:, 1]  # nxm
        right = bboxes[:, 2] - xs  # mxn
        bottom = bboxes[:, 3] - ys

        reg_targets_per_im = tf.stack([left, top, right, bottom], axis=2)  # nxmx4

        is_in_boxes = tf.cast(tf.greater(tf.reduce_min(reg_targets_per_im, axis=2), 0), tf.int32)

        max_reg_targets_per_im = tf.reduce_max(reg_targets_per_im, axis=2)

        is_cared_in_the_level = tf.math.logical_and(tf.greater_equal(max_reg_targets_per_im,
                                                                     tf.reshape(object_sizes_of_interest[:, 0],
                                                                                [-1, 1]) * tf.ones(
                                                                         (1, tf.shape(max_reg_targets_per_im)[1]))),
                                                    tf.less_equal(max_reg_targets_per_im,
                                                                  tf.reshape(object_sizes_of_interest[:, 1],
                                                                             [-1, 1]) * tf.ones(
                                                                      (1, tf.shape(max_reg_targets_per_im)[1]))))
        is_cared_in_the_level = tf.cast(is_cared_in_the_level, tf.int32)

        locations_to_gt_area = tf.ones((tf.shape(xs)[0], tf.shape(area)[0]), dtype=tf.float32) * area  # nxm

        locations_to_gt_area = tf.where(tf.equal(is_in_boxes, 0),
                                        INF * tf.ones_like(locations_to_gt_area),
                                        locations_to_gt_area)

        locations_to_gt_area = tf.where(tf.equal(is_cared_in_the_level, 0),
                                        INF * tf.ones_like(locations_to_gt_area),
                                        locations_to_gt_area)

        locations_to_min_area = tf.reduce_min(locations_to_gt_area, axis=1)
        locations_to_gt_inds = tf.argmin(locations_to_gt_area, axis=1)

        indices = tf.stack([tf.to_int32(tf.range(tf.shape(xs)[0])), tf.to_int32(locations_to_gt_inds)], axis=1)
        reg_targets_per_im = tf.gather_nd(reg_targets_per_im, indices)
        labels_per_im = tf.gather_nd(tf.reshape(tf.ones((tf.shape(xs)[0],), tf.int64), [-1, 1])
                                     * tf.reshape(label, [1, -1]),
                                     indices)
        labels_per_im = tf.where(tf.equal(locations_to_min_area, INF),
                                 tf.zeros_like(labels_per_im),
                                 labels_per_im)

        labels.append(labels_per_im)
        reg_targets.append(reg_targets_per_im)

    return labels, reg_targets


def compute_centerness_targets(reg_targets):
    left_right = tf.gather(reg_targets, [0, 2], axis=1)
    top_bottom = tf.gather(reg_targets, [1, 3], axis=1)
    left_right_min = tf.reduce_min(left_right, axis=1)
    left_right_max = tf.reduce_max(left_right, axis=1)
    top_bottom_min = tf.reduce_min(top_bottom, axis=1)
    top_bottom_max = tf.reduce_max(top_bottom, axis=1)
    centerness = tf.multiply(tf.divide(left_right_min, left_right_max), tf.divide(top_bottom_min, top_bottom_max))
    centerness = tf.sqrt(centerness)
    return tf.stop_gradient(centerness)


def compute_loss(good, labels_flatten, box_cls_flatten,
                 reg_targets_flatten, box_reg_flatten, centerness_preds,
                 num_pos_avg_per_gpu):
    labels_flatten = tf.stop_gradient(labels_flatten)
    reg_targets_flatten = tf.stop_gradient(reg_targets_flatten)

    if good:
        cls_loss = sigmoid_focal_loss4(labels=labels_flatten, logits=box_cls_flatten)
        cls_loss = tf.truediv(cls_loss, num_pos_avg_per_gpu)

        centerness_targets = compute_centerness_targets(reg_targets_flatten)
        centerness_targets_sum = tf.reduce_sum(centerness_targets, name='centerness_targets_sum')
        centerness_targets_sum = tf.stop_gradient(centerness_targets_sum)

        if cfg.TRAIN.NUM_GPUS > 1:
            shared_name = 'centerness_targets_sum'
            sum_centerness_targets_avg_per_gpu = gen_nccl_ops.nccl_all_reduce(
                input=centerness_targets_sum,
                reduction='sum',
                num_devices=cfg.TRAIN.NUM_GPUS,
                shared_name=shared_name) * (1. / cfg.TRAIN.NUM_GPUS)
        else:
            sum_centerness_targets_avg_per_gpu = centerness_targets_sum
        sum_centerness_targets_avg_per_gpu = tf.stop_gradient(sum_centerness_targets_avg_per_gpu)

        # 使用 iou loss
        reg_loss = tf.cond(tf.greater(centerness_targets_sum, 0),
                           lambda: iou_loss(gt_boxes=reg_targets_flatten,
                                                 pred_boxes=box_reg_flatten,
                                                 weights=centerness_targets,
                                                 loss_type=cfg.FCOS.IOU_LOSS_TYPE),
                           lambda: iou_loss(gt_boxes=reg_targets_flatten,
                                                 pred_boxes=box_reg_flatten,
                                                 weights=tf.ones_like(centerness_targets),
                                                 loss_type=cfg.FCOS.IOU_LOSS_TYPE))
        reg_loss = tf.truediv(reg_loss, sum_centerness_targets_avg_per_gpu)

        centerness_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=centerness_targets,
                                                                  logits=centerness_preds)
        centerness_loss = tf.truediv(tf.reduce_sum(centerness_loss), num_pos_avg_per_gpu)
    else:
        cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_flatten,
                                                                  logits=box_cls_flatten)
        cls_loss = tf.reduce_mean(cls_loss)
        reg_loss = tf.reduce_sum(box_reg_flatten)
        centerness_loss = tf.zeros([], dtype=tf.float32)

    return tf.reshape(cls_loss, []), \
           tf.reshape(reg_loss, []), \
           tf.reshape(centerness_loss, [])


def iou_loss(gt_boxes, pred_boxes, weights, loss_type='giou'):
    pred_boxes = tf.cast(pred_boxes, tf.float32)
    gt_boxes = tf.stop_gradient(tf.cast(gt_boxes, tf.float32))
    weights = tf.stop_gradient(tf.cast(weights, tf.float32))

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
    loss = loss * tf.reshape(weights, tf.shape(loss))
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
    term1 = tf.pow(1 - p, gamma) * tf.math.log(p)
    term2 = tf.pow(p, gamma) * tf.math.log(1 - p)
    pos_t = tf.cast(tf.greater_equal(t, 0), tf.float32)
    loss = -pos1 * term1 * alpha - ((1 - pos1) * pos_t) * term2 * (1 - alpha)
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


# NHWC
def output_predictions_for_single_map_batched(location, box_cls, box_reg, centerness):
    shape = tf.shape(box_cls)
    N, H, W, C = shape[0], shape[1], shape[2], shape[3]
    box_cls = tf.sigmoid(tf.reshape(box_cls, [N, -1, C]))
    box_reg = tf.reshape(box_reg, [N, -1, 4])
    centerness = tf.sigmoid(tf.reshape(centerness, [N, -1]))

    candidate_inds = tf.cast(tf.greater(box_cls, cfg.FCOS.PRE_NMS_THRESH), tf.int32)  # N*(HW)*C
    pre_nms_top_n = tf.reduce_sum(tf.reshape(candidate_inds, [N, -1]), axis=1)
    pre_nms_top_n = tf.where(tf.greater(pre_nms_top_n, cfg.FCOS.PRE_NMS_TOP_N),
                             cfg.FCOS.PRE_NMS_TOP_N * tf.ones_like(pre_nms_top_n, dtype=pre_nms_top_n.dtype),
                             pre_nms_top_n)
    box_cls = tf.multiply(box_cls, tf.tile(tf.expand_dims(centerness, axis=2), [1, 1, C]))  # N*(HW)*C

    def condition(b_id, batch_pred_boxes, batch_pred_scores, batch_pred_labels, batch_inds):
        return tf.less(b_id, N)

    def body(b_id, batch_pred_boxes, batch_pred_scores, batch_pred_labels, batch_inds):
        per_pre_nms_top_n = pre_nms_top_n[b_id]
        per_box_cls = box_cls[b_id]  # HW*C
        per_box_regression = box_reg[b_id]  # HW*4
        per_candidate_inds = candidate_inds[b_id]  # HW*C

        def true_fn(per_box_cls_, per_box_regression_):
            valid_inds = tf.where(tf.greater(per_candidate_inds, 0))
            per_box_cls_ = tf.gather_nd(per_box_cls_, valid_inds)
            per_box_loc, per_class = tf.split(valid_inds, num_or_size_splits=2, axis=1)
            per_class = tf.add(per_class, 1)
            per_box_loc = tf.reshape(per_box_loc, [-1])
            per_box_regression_ = tf.gather(per_box_regression_, per_box_loc)
            per_location = tf.gather(location, per_box_loc)

            per_box_cls_, top_k_indices = tf.nn.top_k(per_box_cls_, k=per_pre_nms_top_n, sorted=False)
            per_class = tf.gather(per_class, top_k_indices)
            per_box_regression_ = tf.gather(per_box_regression_, top_k_indices)
            per_location = tf.gather(per_location, top_k_indices)

            xs, ys = tf.split(per_location, num_or_size_splits=2, axis=1)
            left, top, right, bottom = tf.split(per_box_regression_, num_or_size_splits=4, axis=1)
            left = xs - left
            top = ys - top
            right = right + xs
            bottom = bottom + ys
            per_box_regression_ = tf.concat([left, top, right, bottom], axis=1)

            return tf.reshape(per_box_regression_, [-1, 4]), \
                   tf.reshape(tf.sqrt(per_box_cls_), [-1]), \
                   tf.reshape(per_class, [-1])

        def false_fn(per_box_cls_, per_box_regression_):
            valid_inds = tf.where(tf.greater(per_candidate_inds, 0))
            per_box_cls_ = tf.gather_nd(per_box_cls_, valid_inds)
            per_box_loc, per_class = tf.split(valid_inds, num_or_size_splits=2, axis=1)
            per_class = tf.add(per_class, 1)
            per_box_loc = tf.reshape(per_box_loc, [-1])
            per_box_regression_ = tf.gather(per_box_regression_, per_box_loc)
            per_location = tf.gather(location, per_box_loc)

            xs, ys = tf.split(per_location, num_or_size_splits=2, axis=1)
            left, top, right, bottom = tf.split(per_box_regression_, num_or_size_splits=4, axis=1)
            left = xs - left
            top = ys - top
            right = right + xs
            bottom = bottom + ys
            per_box_regression_ = tf.concat([left, top, right, bottom], axis=1)

            return tf.reshape(per_box_regression_, [-1, 4]), \
                   tf.reshape(tf.sqrt(per_box_cls_), [-1]), \
                   tf.reshape(per_class, [-1])

        current_batch_boxes, current_batch_scores, current_batch_labels = \
            tf.cond(tf.greater(tf.reduce_sum(per_candidate_inds), per_pre_nms_top_n),
                    lambda: true_fn(per_box_cls, per_box_regression),
                    lambda: false_fn(per_box_cls, per_box_regression))

        current_batch_inds = b_id * tf.ones((tf.shape(current_batch_boxes)[0],), dtype=tf.int32)

        batch_pred_boxes = tf.cond(tf.equal(b_id, 0),
                                   lambda: current_batch_boxes,
                                   lambda: tf.concat([batch_pred_boxes, current_batch_boxes], axis=0))
        batch_pred_scores = tf.cond(tf.equal(b_id, 0),
                                    lambda: current_batch_scores,
                                    lambda: tf.concat([batch_pred_scores, current_batch_scores], axis=0))
        batch_pred_labels = tf.cond(tf.equal(b_id, 0),
                                    lambda: current_batch_labels,
                                    lambda: tf.concat([batch_pred_labels, current_batch_labels], axis=0))
        batch_inds = tf.cond(tf.equal(b_id, 0),
                             lambda: current_batch_inds,
                             lambda: tf.concat([batch_inds, current_batch_inds], axis=0))

        return tf.add(b_id, 1), batch_pred_boxes, batch_pred_scores, batch_pred_labels, batch_inds

    batch_pred_boxes = tf.zeros([0, 4], dtype=tf.float32)
    batch_pred_scores = tf.zeros([0, ], dtype=tf.float32)
    batch_pred_labels = tf.zeros([0, ], dtype=tf.int64)
    batch_inds = tf.zeros([0, ], dtype=tf.int32)
    b_id = tf.constant(0, dtype=tf.int32)
    index_results = (b_id, batch_pred_boxes, batch_pred_scores, batch_pred_labels, batch_inds)
    _, final_boxes, final_scores, final_category, final_inds = \
        tf.while_loop(condition, body, index_results,
                      shape_invariants=(b_id.get_shape(),
                                        tf.TensorShape([None, 4]),
                                        tf.TensorShape([None]),
                                        tf.TensorShape([None]),
                                        tf.TensorShape([None])))

    return final_boxes, final_scores, final_category, final_inds


# NHWC
def output_predictions_for_single_map(location, box_cls, box_reg, centerness):
    shape = tf.shape(box_cls)
    N, H, W, C = shape[0], shape[1], shape[2], shape[3]
    box_cls = tf.sigmoid(tf.reshape(box_cls, [N, -1, C]))
    box_reg = tf.reshape(box_reg, [N, -1, 4])
    centerness = tf.sigmoid(tf.reshape(centerness, [N, -1]))

    candidate_inds = tf.cast(tf.greater(box_cls, cfg.FCOS.PRE_NMS_THRESH), tf.int32)  # N*(HW)*C
    pre_nms_top_n = tf.reduce_sum(tf.reshape(candidate_inds, [N, -1]), axis=1)
    pre_nms_top_n = tf.where(tf.greater(pre_nms_top_n, cfg.FCOS.PRE_NMS_TOP_N),
                             cfg.FCOS.PRE_NMS_TOP_N * tf.ones_like(pre_nms_top_n, dtype=pre_nms_top_n.dtype),
                             pre_nms_top_n)
    box_cls = tf.multiply(box_cls, tf.tile(tf.expand_dims(centerness, axis=2), [1, 1, C]))  # N*(HW)*C

    per_pre_nms_top_n = pre_nms_top_n[0]
    per_box_cls = box_cls[0]  # HW*C
    per_box_regression = box_reg[0]  # HW*4
    per_candidate_inds = candidate_inds[0]  # HW*C

    def true_fn(per_box_cls_, per_box_regression_):
        valid_inds = tf.where(tf.greater(per_candidate_inds, 0))
        per_box_cls_ = tf.gather_nd(per_box_cls_, valid_inds)
        per_box_loc, per_class = tf.split(valid_inds, num_or_size_splits=2, axis=1)
        per_class = tf.add(per_class, 1)
        per_box_loc = tf.reshape(per_box_loc, [-1])
        per_box_regression_ = tf.gather(per_box_regression_, per_box_loc)
        per_location = tf.gather(location, per_box_loc)

        per_box_cls_, top_k_indices = tf.nn.top_k(per_box_cls_, k=per_pre_nms_top_n, sorted=False)
        per_class = tf.gather(per_class, top_k_indices)
        per_box_regression_ = tf.gather(per_box_regression_, top_k_indices)
        per_location = tf.gather(per_location, top_k_indices)

        xs, ys = tf.split(per_location, num_or_size_splits=2, axis=1)
        left, top, right, bottom = tf.split(per_box_regression_, num_or_size_splits=4, axis=1)
        left = xs - left
        top = ys - top
        right = right + xs
        bottom = bottom + ys
        per_box_regression_ = tf.concat([left, top, right, bottom], axis=1)

        return tf.reshape(per_box_regression_, [-1, 4]), \
               tf.reshape(tf.sqrt(per_box_cls_), [-1]), \
               tf.reshape(per_class, [-1])

    def false_fn(per_box_cls_, per_box_regression_):
        valid_inds = tf.where(tf.greater(per_candidate_inds, 0))
        per_box_cls_ = tf.gather_nd(per_box_cls_, valid_inds)
        per_box_loc, per_class = tf.split(valid_inds, num_or_size_splits=2, axis=1)
        per_class = tf.add(per_class, 1)
        per_box_loc = tf.reshape(per_box_loc, [-1])
        per_box_regression_ = tf.gather(per_box_regression_, per_box_loc)
        per_location = tf.gather(location, per_box_loc)

        xs, ys = tf.split(per_location, num_or_size_splits=2, axis=1)
        left, top, right, bottom = tf.split(per_box_regression_, num_or_size_splits=4, axis=1)
        left = xs - left
        top = ys - top
        right = right + xs
        bottom = bottom + ys
        per_box_regression_ = tf.concat([left, top, right, bottom], axis=1)

        return tf.reshape(per_box_regression_, [-1, 4]), \
               tf.reshape(tf.sqrt(per_box_cls_), [-1]), \
               tf.reshape(per_class, [-1])

    final_boxes, final_scores, final_category = \
        tf.cond(tf.greater(tf.reduce_sum(per_candidate_inds), per_pre_nms_top_n),
                lambda: true_fn(per_box_cls, per_box_regression),
                lambda: false_fn(per_box_cls, per_box_regression))

    final_inds = tf.zeros((tf.shape(final_boxes)[0],), dtype=tf.int32)

    return final_boxes, final_scores, final_category, final_inds


# NHWC
def output_predictions(locations, fcos_outputs):
    sampled_boxes = []
    for level in range(len(cfg.FCOS.FPN_STRIDES)):
        box_cls = fcos_outputs[level][0]
        box_reg = fcos_outputs[level][1]
        centerness = fcos_outputs[level][2]
        sampled_boxes.append(
            output_predictions_for_single_map(
                locations[level], box_cls, box_reg, centerness)
        )
    return sampled_boxes


from resnet_model import group_normalization, resnet_v1_fcos_backbone


def fcos_head(features, num_channels=256, num_classes=80, is_training=False, data_format='channels_last',
              reuse=None):

    kernel_init = tf.random_normal_initializer(stddev=0.01)
    bias_init = tf.constant_initializer(-math.log((1 - cfg.FCOS.PRIOR_PROB) / cfg.FCOS.PRIOR_PROB))

    with tf.variable_scope('fcos', reuse=reuse):
        outputs = []
        for level, feature in enumerate(features):
            scale = tf.get_variable(name='scale%d'%(level+3), shape=[], initializer=tf.constant_initializer(1.0))

            with tf.variable_scope('cls_branch', reuse=tf.AUTO_REUSE):
                l = feature
                for i in range(4):
                    l = tf.layers.conv2d(l, filters=num_channels, kernel_size=3, padding='same',
                                         data_format=data_format, use_bias=True, kernel_initializer=kernel_init,
                                         name='conv%d'%i)
                    l = group_normalization(l, group=32, data_format=data_format, name='gn%d'%i)
                    l = tf.nn.relu(l)
                cls_tower = l

            with tf.variable_scope('reg_branch', reuse=tf.AUTO_REUSE):
                l = feature
                for i in range(4):
                    l = tf.layers.conv2d(l, filters=num_channels, kernel_size=3, padding='same',
                                         data_format=data_format, use_bias=True, kernel_initializer=kernel_init,
                                         name='conv%d'%i)
                    l = group_normalization(l, group=32, data_format=data_format,  name='gn%d'%i)
                    l = tf.nn.relu(l)
                bbox_tower = l

            with tf.variable_scope('output', reuse=tf.AUTO_REUSE):
                cls_logits = tf.layers.conv2d(cls_tower, filters=num_classes, kernel_size=3,
                                              padding='same', name='cls', data_format=data_format,
                                              kernel_initializer=kernel_init, bias_initializer=bias_init)
                bbox_pred = tf.layers.conv2d(bbox_tower, filters=4, kernel_size=3,
                                             padding='same', name='bbox', data_format=data_format,
                                             kernel_initializer=kernel_init)

                if cfg.FCOS.CENTERNESS_ON_REG:
                    centerness_logits = tf.layers.conv2d(bbox_tower, filters=1, kernel_size=3,
                                                         padding='same', name='centerness',
                                                         data_format=data_format, kernel_initializer=kernel_init)

                else:
                    centerness_logits = tf.layers.conv2d(cls_tower, filters=1, kernel_size=3,
                                                       padding='same', name='centerness',
                                                       data_format=data_format, kernel_initializer=kernel_init)

                bbox_pred = scale * bbox_pred  # N4HW
                if cfg.FCOS.NORM_REG_TARGETS:
                    bbox_pred = tf.nn.relu(bbox_pred)
                else:
                    bbox_pred = tf.exp(bbox_pred)
                outputs.append((cls_logits, bbox_pred, centerness_logits))

        return outputs


def inference(locations, fcos_outputs):
    if cfg.FCOS.NORM_REG_TARGETS:
        for level in range(len(cfg.FCOS.FPN_STRIDES)):
            fcos_outputs[level][1] *= cfg.FCOS.FPN_STRIDES[level]

    pred_results = output_predictions(locations, fcos_outputs)

    final_boxes = []
    final_scores = []
    final_labels = []
    for level in range(len(cfg.FCOS.FPN_STRIDES)):
        tf.identity(fcos_outputs[level][0], name='outputs/P{}_box_cls_logits'.format(level + 3))
        tf.identity(fcos_outputs[level][1], name='outputs/P{}_box_reg_logits'.format(level + 3))
        tf.identity(fcos_outputs[level][2], name='outputs/P{}_centerness'.format(level + 3))

        tf.identity(pred_results[level][0], name='outputs/P{}_boxes'.format(level + 3))
        tf.identity(pred_results[level][1], name='outputs/P{}_scores'.format(level + 3))
        tf.identity(pred_results[level][2], name='outputs/P{}_labels'.format(level + 3))
        tf.identity(pred_results[level][3], name='outputs/P{}_indices'.format(level + 3))

        final_boxes.append(pred_results[level][0])
        final_scores.append(pred_results[level][1])
        final_labels.append(pred_results[level][2])
    final_boxes = tf.concat(final_boxes, axis=0, name='output/boxes')
    final_scores = tf.concat(final_scores, axis=0, name='output/scores')
    final_labels = tf.concat(final_labels, axis=0, name='output/labels')

    return final_boxes, final_scores, final_labels


def loss(images, locations, fcos_outputs, gt_boxes, gt_labels, orig_gt_counts, data_format):
    ###################################################################################
    # visualization
    if cfg.FCOS.VISUALIZATION:
        print('visualization ')
        with tf.device('/cpu:0'):
            with tf.name_scope('vis_gt'):
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
                    if TF_version <= (1, 14):
                        tf.summary.image('gt_img_{}'.format(b_id), gt_img)
                    else:
                        tf.compat.v1.summary.image('gt_img_{}'.format(b_id), gt_img)
    ###################################################################################

    if cfg.FCOS.VISUALIZATION:
        print('visualization ')
        with tf.device('/cpu:0'):
            with tf.name_scope('vis_pred'):
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
                shp2d = tf.shape(images_show)[1:3]
                img_h, img_w = shp2d[0], shp2d[1]

                for level in range(len(cfg.FCOS.FPN_STRIDES)):
                    with tf.name_scope('P_{}'.format(level + 3)):
                        locations_this_level = locations[level]  # nx2
                        labels_this_level = tf.transpose(fcos_outputs[level][0], [0, 2, 3, 1])
                        reg_targets_this_level = tf.transpose(fcos_outputs[level][1], [0, 2, 3, 1])
                        centerness_this_level = tf.transpose(fcos_outputs[level][2], [0, 2, 3, 1])
                        centerness_this_level = tf.sigmoid(centerness_this_level)
                        feat_shp2d = tf.shape(centerness_this_level)[1:3]
                        feat_h, feat_w = feat_shp2d[0], feat_shp2d[1]

                        # centerness
                        if TF_version <= (1, 14):
                            tf.summary.image('centerness_{}'.format(level + 3), centerness_this_level)
                        else:
                            tf.compat.v1.summary.image('centerness_{}'.format(level + 3), centerness_this_level)

                        # box_reg
                        left, top, right, bottom = tf.split(reg_targets_this_level, num_or_size_splits=4,
                                                            axis=3)
                        temp = {'left': left, 'right': right, 'top': top, 'bottom': bottom}
                        for key in temp.keys():
                            target_im_this_level = tf.reshape(temp[key],
                                                              [cfg.TRAIN.BATCH_SIZE_PER_GPU, feat_h, feat_w, 1])
                            if TF_version <= (1, 14):
                                tf.summary.image('reg_pred_{}'.format(key), target_im_this_level)
                            else:
                                tf.compat.v1.summary.image('reg_pred_{}'.format(key), target_im_this_level)

    images = tf.stop_gradient(images)
    locations = [tf.stop_gradient(location) for location in locations]
    gt_boxes = tf.stop_gradient(gt_boxes)
    gt_labels = tf.stop_gradient(gt_labels)
    orig_gt_counts = tf.stop_gradient(orig_gt_counts)

    labels, reg_targets = prepare_targets(images, locations, gt_boxes, gt_labels, orig_gt_counts)

    # class, bbox_reg, centerness
    box_cls_flatten = []
    box_reg_flatten = []
    centerness_flatten = []
    labels_flatten = []
    reg_targets_flatten = []
    for l in range(len(cfg.FCOS.FPN_STRIDES)):
        box_cls_flatten.append(tf.reshape(tf.transpose(fcos_outputs[l][0], [0, 2, 3, 1]),
                                          [-1, cfg.DATA.NUM_CATEGORY]))
        box_reg_flatten.append(tf.reshape(tf.transpose(fcos_outputs[l][1], [0, 2, 3, 1]),
                                          [-1, 4]))
        centerness_flatten.append(tf.reshape(tf.transpose(fcos_outputs[l][2], [0, 2, 3, 1]),
                                             [-1]))
        labels_flatten.append(tf.reshape(labels[l], [-1]))
        reg_targets_flatten.append(tf.reshape(reg_targets[l], [-1, 4]))

    box_cls_flatten = tf.concat(box_cls_flatten, axis=0)
    box_reg_flatten = tf.concat(box_reg_flatten, axis=0)
    centerness_flatten = tf.concat(centerness_flatten, axis=0)
    labels_flatten = tf.concat(labels_flatten, axis=0)
    reg_targets_flatten = tf.concat(reg_targets_flatten, axis=0)

    labels_flatten = tf.stop_gradient(labels_flatten)
    reg_targets_flatten = tf.stop_gradient(reg_targets_flatten)

    # 只计算有框的回归
    pos_inds = tf.reshape(tf.where(tf.greater(labels_flatten, 0)), [-1])
    box_reg_flatten = tf.gather(box_reg_flatten, pos_inds)
    centerness_flatten = tf.gather(centerness_flatten, pos_inds)
    reg_targets_flatten = tf.gather(reg_targets_flatten, pos_inds)

    num_pos_inds = tf.cond(tf.greater(tf.size(pos_inds), 0),
                           lambda: tf.shape(pos_inds)[0],
                           lambda: 0)
    num_pos_inds = tf.cast(num_pos_inds, tf.float32, name='num_pos_inds')
    num_pos_inds = tf.stop_gradient(num_pos_inds)

    if cfg.TRAIN.NUM_GPUS > 1:
        shared_name = 'num_pos_inds'  # re.sub('tower[0-9]+/', '', tf.get_variable_scope().name)
        num_pos_inds_sum = gen_nccl_ops.nccl_all_reduce(
            input=num_pos_inds,
            reduction='sum',
            num_devices=cfg.TRAIN.NUM_GPUS,
            shared_name=shared_name) * (1. / cfg.TRAIN.NUM_GPUS)
        num_pos_avg_per_gpu = tf.maximum(num_pos_inds_sum, 1.0)
    else:
        num_pos_avg_per_gpu = num_pos_inds
    num_pos_avg_per_gpu = tf.stop_gradient(num_pos_avg_per_gpu)

    # 回归损失和centerness损失
    cls_loss, reg_loss, centerness_loss \
        = tf.cond(tf.greater(tf.cast(num_pos_inds, tf.int32), 0),
                  lambda: compute_loss(True, labels_flatten, box_cls_flatten,
                                       reg_targets_flatten, box_reg_flatten,
                                       centerness_flatten, num_pos_avg_per_gpu),
                  lambda: compute_loss(False, labels_flatten, box_cls_flatten,
                                       reg_targets_flatten, box_reg_flatten,
                                       centerness_flatten, num_pos_avg_per_gpu))

    loss_dict = {'cls_loss': cls_loss,
                 'reg_loss': reg_loss,
                 'centerness_loss': centerness_loss}

    return loss_dict


def model(inputs, is_training=False, reuse=None, data_format='channels_last', mode='train'):

    if mode == 'train':
        images, gt_boxes, gt_labels, orig_gt_counts = inputs
    else:
        images = inputs

    with tf.variable_scope('resnet50', reuse=reuse):
        p34567 = resnet_v1_fcos_backbone(images, resnet_depth=50, is_training=is_training, data_format=data_format)

    locations = compute_locations(p34567, data_format=data_format)

    fcos_outputs = fcos_head(p34567, num_channels=256, data_format=data_format, reuse=reuse)

    if mode == 'train':
        loss_dict = loss(images, locations, fcos_outputs, gt_boxes, gt_labels, orig_gt_counts, data_format=data_format)
        return loss_dict
    else:
        return inference(locations, fcos_outputs)


if __name__ == '__main__':
    import cv2
    import importlib.machinery

    loader = importlib.machinery.SourceFileLoader('caffepb', 'caffe_pb2.py')  # protoc caffe.proto --python_out .
    obj = loader.load_module().BlobProto()
    obj.ParseFromString(open('ResNet_mean.binaryproto', 'rb').read())
    pp_mean_224 = np.array(obj.data).reshape(3, 224, 224).transpose(1, 2, 0)  # 224, 224, 3

    IMAGE_MEAN_BGR = [103.530, 116.280, 123.675]
    imgfilename = 'grace_hopper.jpg'
    im = cv2.imread(imgfilename).astype('float32')  # bgr
    # im -= np.asarray(IMAGE_MEAN_BGR)
    h, w = im.shape[:2]
    scale = 256 * 1.0 / min(h, w)
    if h < w:
        newh, neww = 256, int(scale * w + 0.5)
    else:
        newh, neww = int(scale * h + 0.5), 256
    im = cv2.resize(im, (neww, newh))
    orig_shape = im.shape
    assert orig_shape[0] >= 224 \
           and orig_shape[1] >= 224, orig_shape
    h0 = int((orig_shape[0] - 224) * 0.5)
    w0 = int((orig_shape[1] - 224) * 0.5)
    im = im[h0:h0 + 224, w0:w0 + 224]
    im = np.reshape(im - pp_mean_224, (1, 224, 224, 3))  # 652, 0.85
    # im = np.reshape(im, (1, 224, 224, 3))       # 652, 0.80

    ckpt_params = dict(np.load('MSRA-R50.npz'))

    data_format = 'channels_last'

    if data_format == 'channels_first':
        im = np.transpose(im, [0, 3, 1, 2])

    image_pl = tf.constant(im.astype(np.float32))

    # if data_format == 'channels_first':
    #     image_pl = tf.placeholder(tf.float32, [None, 3, None, None])
    # else:
    #     image_pl = tf.placeholder(tf.float32, [None, None, None, 3])

    logits = resnet_v1(image_pl, is_training=False, data_format=data_format)
    axis = 3 if data_format == 'channels_last' else 1
    probs = tf.nn.softmax(logits, axis=axis)

    init_op = tf.group([tf.global_variables_initializer(),
                        tf.local_variables_initializer()])

    # print('='*50)
    # for var in tf.trainable_variables():
    #     print(var.name, var.shape)
    # print('='*50)
    # for k,v in ckpt_params.items():
    #     print(k, v.shape)

    for var in tf.global_variables():
        dst_name = var.name
        print(var.name, var.shape)

    print('load weights ...')
    assign_ops = []
    all_variables = []
    for var in tf.global_variables():
        dst_name = var.name
        all_variables.append(dst_name + '\n')
        if 'resnet50' in dst_name:
            src_name = dst_name.replace('resnet50/', ''). \
                replace('conv2d/kernel:0', 'W') \
                .replace('conv2d/bias:0', 'b') \
                .replace('batch_normalization/gamma:0', 'gamma') \
                .replace('batch_normalization/beta:0', 'beta') \
                .replace('batch_normalization/moving_mean:0', 'mean/EMA') \
                .replace('batch_normalization/moving_variance:0', 'variance/EMA') \
                .replace('kernel:0', 'W').replace('bias:0', 'b')
            if 'batch_normalization' in dst_name:
                src_name = src_name.replace('res', 'bn')
                if 'conv1' in src_name:
                    src_name = 'bn_' + src_name
            if dst_name == 'resnet50/conv1000/kernel:0':
                print('{} --> {} {}'.format('fc1000/W', dst_name, var.shape))
                assign_ops.append(tf.assign(var, ckpt_params['fc1000/W']))
                continue
            if src_name in ckpt_params:
                print('{} --> {} {}'.format(src_name, dst_name, var.shape))
                assign_ops.append(tf.assign(var, ckpt_params[src_name]))
    print('load weights done.')
    with open('all_variables.txt', 'w') as fp:
        fp.writelines(all_variables)

    sess = tf.Session()
    sess.run(init_op)
    sess.run(assign_ops)

    if False:

        IMAGE_MEAN_BGR = [103.530, 116.280, 123.675]
        imgfilename = 'grace_hopper.jpg'
        im = cv2.imread(imgfilename).astype('float32')  # bgr
        im -= np.asarray(IMAGE_MEAN_BGR)
        h, w = im.shape[:2]
        short_side_len = 800
        scale = short_side_len * 1.0 / min(h, w)
        if h < w:
            newh, neww = short_side_len, int(scale * w + 0.5)
        else:
            newh, neww = int(scale * h + 0.5), short_side_len
        im = cv2.resize(im, (neww, newh))

        h, w = im.shape[:2]
        mult = float(cfg.FPN.RESOLUTION_REQUIREMENT)  # size divisable
        max_height = int(np.ceil(float(h) / mult) * mult)
        max_width = int(np.ceil(float(w) / mult) * mult)
        im1 = np.zeros((1, max_height, max_width, 3), dtype=np.float32)
        im1[:, :h, :w, :] = im

        logits_, probs_ = sess.run([logits, probs], feed_dict={image_pl: im1})
        print(probs_.shape)
        max_probs = np.max(probs_[0], axis=-1)
        maxprobs_map = cv2.resize(max_probs, im1.shape[1:3][::-1])
        cv2.imwrite('maxprobs_map.png', (maxprobs_map * 255).astype(np.uint8))
        im1 = im1[0] + np.asarray(IMAGE_MEAN_BGR)
        cv2.imwrite('im1.png', im1.astype(np.uint8))

        print('max_probs', max_probs)
        maxids = np.argmax(probs_[0], axis=-1)
        print('maxids', maxids)
        pos = np.argmax(max_probs)
        pos1 = np.unravel_index([pos], shape=probs_.shape[1:3])
        maxid = maxids[pos1]
        print(max_probs[pos1], maxid)

    else:
        logits_, probs_ = sess.run([logits, probs])
        print(probs_.shape)
        maxid = np.argmax(probs_)

        if data_format == 'channels_first':
            print(maxid, probs_[0, maxid, 0, 0])
        else:
            print(maxid, probs_[0, 0, 0, maxid])


def backup():
    import cv2
    import importlib.machinery

    ckpt_params = dict(np.load('MSRA-R50.npz'))

    data_format = 'channels_first'

    if data_format == 'channels_first':
        image_pl = tf.placeholder(tf.float32, [None, 3, None, None])
    else:
        image_pl = tf.placeholder(tf.float32, [None, None, None, 3])
    logits = resnet_v1(image_pl, is_training=False, data_format=data_format)
    probs = tf.nn.softmax(logits)

    init_op = tf.group([tf.global_variables_initializer(),
                        tf.local_variables_initializer()])

    # print('='*50)
    # for var in tf.trainable_variables():
    #     print(var.name, var.shape)
    # print('='*50)
    # for k,v in ckpt_params.items():
    #     print(k, v.shape)

    for var in tf.global_variables():
        dst_name = var.name
        print(var.name, var.shape)

    print('load weights ...')
    assign_ops = []
    all_variables = []
    for var in tf.global_variables():
        dst_name = var.name
        all_variables.append(dst_name + '\n')
        if 'resnet50' in dst_name:
            src_name = dst_name.replace('resnet50/', ''). \
                replace('conv2d/kernel:0', 'W') \
                .replace('conv2d/bias:0', 'b') \
                .replace('batch_normalization/gamma:0', 'gamma') \
                .replace('batch_normalization/beta:0', 'beta') \
                .replace('batch_normalization/moving_mean:0', 'mean/EMA') \
                .replace('batch_normalization/moving_variance:0', 'variance/EMA') \
                .replace('kernel:0', 'W').replace('bias:0', 'b')
            if 'batch_normalization' in dst_name:
                src_name = src_name.replace('res', 'bn')
                if 'conv1' in src_name:
                    src_name = 'bn_' + src_name
            if dst_name == 'resnet50/conv1000/kernel:0':
                print('{} --> {} {}'.format('fc1000/W', dst_name, var.shape))
                assign_ops.append(tf.assign(var, ckpt_params['fc1000/W']))
                continue
            if src_name in ckpt_params:
                print('{} --> {} {}'.format(src_name, dst_name, var.shape))
                assign_ops.append(tf.assign(var, ckpt_params[src_name]))
    print('load weights done.')
    with open('all_variables.txt', 'w') as fp:
        fp.writelines(all_variables)

    sess = tf.Session()
    sess.run(init_op)
    sess.run(assign_ops)

    if False:

        IMAGE_MEAN_BGR = [103.530, 116.280, 123.675]
        imgfilename = 'grace_hopper.jpg'
        im = cv2.imread(imgfilename).astype('float32')  # bgr
        im -= np.asarray(IMAGE_MEAN_BGR)
        h, w = im.shape[:2]
        short_side_len = 800
        scale = short_side_len * 1.0 / min(h, w)
        if h < w:
            newh, neww = short_side_len, int(scale * w + 0.5)
        else:
            newh, neww = int(scale * h + 0.5), short_side_len
        im = cv2.resize(im, (neww, newh))

        h, w = im.shape[:2]
        mult = float(cfg.FPN.RESOLUTION_REQUIREMENT)  # size divisable
        max_height = int(np.ceil(float(h) / mult) * mult)
        max_width = int(np.ceil(float(w) / mult) * mult)
        im1 = np.zeros((1, max_height, max_width, 3), dtype=np.float32)
        im1[:, :h, :w, :] = im

        logits_, probs_ = sess.run([logits, probs], feed_dict={image_pl: im1})
        print(probs_.shape)
        max_probs = np.max(probs_[0], axis=-1)
        maxprobs_map = cv2.resize(max_probs, im1.shape[1:3][::-1])
        cv2.imwrite('maxprobs_map.png', (maxprobs_map*255).astype(np.uint8))
        im1 = im1[0] + np.asarray(IMAGE_MEAN_BGR)
        cv2.imwrite('im1.png', im1.astype(np.uint8))

        print('max_probs', max_probs)
        maxids = np.argmax(probs_[0], axis=-1)
        print('maxids', maxids)
        pos = np.argmax(max_probs)
        pos1 = np.unravel_index([pos], shape=probs_.shape[1:3])
        maxid = maxids[pos1]
        print(max_probs[pos1], maxid)

    else:
        import importlib.machinery

        loader = importlib.machinery.SourceFileLoader('caffepb', 'caffe_pb2.py')  # protoc caffe.proto --python_out .
        obj = loader.load_module().BlobProto()
        obj.ParseFromString(open('ResNet_mean.binaryproto', 'rb').read())
        pp_mean_224 = np.array(obj.data).reshape(3, 224, 224).transpose(1, 2, 0)  # 224, 224, 3

        IMAGE_MEAN_BGR = [103.530, 116.280, 123.675]
        imgfilename = 'grace_hopper.jpg'
        im = cv2.imread(imgfilename).astype('float32')  # bgr
        # im -= np.asarray(IMAGE_MEAN_BGR)
        h, w = im.shape[:2]
        scale = 256 * 1.0 / min(h, w)
        if h < w:
            newh, neww = 256, int(scale * w + 0.5)
        else:
            newh, neww = int(scale * h + 0.5), 256
        im = cv2.resize(im, (neww, newh))
        orig_shape = im.shape
        assert orig_shape[0] >= 224 \
               and orig_shape[1] >= 224, orig_shape
        h0 = int((orig_shape[0] - 224) * 0.5)
        w0 = int((orig_shape[1] - 224) * 0.5)
        im = im[h0:h0 + 224, w0:w0 + 224]
        im = np.reshape(im - pp_mean_224, (1, 224, 224, 3))  # 652, 0.85
        # im = np.reshape(im, (1, 224, 224, 3))       # 652, 0.80

        if data_format == 'channels_first':
            im = np.transpose(im, [0, 3, 1, 2])

        logits_, probs_ = sess.run([logits, probs], feed_dict={image_pl: im})
        print(probs_)
        print(probs_.shape)
        maxid = np.argmax(probs_, axis=1)

        if data_format == 'channels_first':
            print(maxid, probs_[0, maxid, 0, 0])
        else:
            print(maxid, probs_[0, 0, 0, maxid])



