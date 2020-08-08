# -*- coding: utf-8 -*-
# @Time    : 2019/11/6 17:27
# @Author  : Zisha Zhong
# @Email   : zhongzisha@outlook.com
# @File    : test_model.py
# @Software: PyCharm


import time
import sys,os
import numpy as np
import tensorflow as tf

from common import warmup_lr_schedule, regularize_cost, \
    l2_regularizer, \
    average_gradients
from config import cfg
from dataset import register_coco
from data import get_train_dataflow
import fasterrcnn_model as model

COMMON_POSTFIX = 'train_FasterRCNN_FPN'


def tower_loss_func(inputs, reuse=False):
    with tf.variable_scope('resnet50', reuse=reuse):
        loss_dict = model.model_fpn(inputs, is_training=False, reuse=reuse,
                                    data_format=cfg.BACKBONE.DATA_FORMAT, mode='train')
    return loss_dict


def train():
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.TRAIN.GPU_LIST
    gpus = list(range(len(cfg.TRAIN.GPU_LIST.split(','))))
    num_gpus = len(gpus)

    restore_from_original_checkpoint = True
    checkpoint_path = cfg.TRAIN.LOG_DIR + COMMON_POSTFIX
    if not tf.io.gfile.exists(checkpoint_path):
        tf.io.gfile.makedirs(checkpoint_path)
    else:
        restore_from_original_checkpoint = False

    register_coco(os.path.expanduser(cfg.DATA.BASEDIR))

    data_iter = get_train_dataflow(batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * num_gpus)
    ds = tf.data.Dataset.from_generator(
        lambda: map(lambda x:
                    tuple([x[k] for k in ['images', 'gt_boxes', 'gt_labels', 'orig_gt_counts',
                                          'all_anchors_level2', 'anchor_labels_level2', 'anchor_boxes_level2',
                                          'all_anchors_level3', 'anchor_labels_level3', 'anchor_boxes_level3',
                                          'all_anchors_level4', 'anchor_labels_level4', 'anchor_boxes_level4',
                                          'all_anchors_level5', 'anchor_labels_level5', 'anchor_boxes_level5',
                                          'all_anchors_level6', 'anchor_labels_level6', 'anchor_boxes_level6']]),
                                       data_iter),
                           (tf.float32, tf.float32, tf.int64, tf.int32,
                            tf.float32, tf.int32, tf.float32,
                            tf.float32, tf.int32, tf.float32,
                            tf.float32, tf.int32, tf.float32,
                            tf.float32, tf.int32, tf.float32,
                            tf.float32, tf.int32, tf.float32),
                           (tf.TensorShape([None, None, None, 3]),
                            tf.TensorShape([None, None, 4]),
                            tf.TensorShape([None, None]),
                            tf.TensorShape([None, ]),
                            tf.TensorShape([None, None, None, None]),
                            tf.TensorShape([None, None, None, None]), tf.TensorShape([None, None, None, None, 4]), #lv2
                            tf.TensorShape([None, None, None, None]),
                            tf.TensorShape([None, None, None, None]), tf.TensorShape([None, None, None, None, 4]), #lv3
                            tf.TensorShape([None, None, None, None]),
                            tf.TensorShape([None, None, None, None]), tf.TensorShape([None, None, None, None, 4]), #lv4
                            tf.TensorShape([None, None, None, None]),
                            tf.TensorShape([None, None, None, None]), tf.TensorShape([None, None, None, None, 4]), #lv5
                            tf.TensorShape([None, None, None, None]),
                            tf.TensorShape([None, None, None, None]), tf.TensorShape([None, None, None, None, 4])  #lv6
                            ))
    ds = ds.prefetch(buffer_size=128)
    ds = ds.make_one_shot_iterator()
    images, gt_boxes, gt_labels, orig_gt_counts, \
    all_anchors_level2, anchor_labels_level2, anchor_boxes_level2, \
    all_anchors_level3, anchor_labels_level3, anchor_boxes_level3, \
    all_anchors_level4, anchor_labels_level4, anchor_boxes_level4, \
    all_anchors_level5, anchor_labels_level5, anchor_boxes_level5, \
    all_anchors_level6, anchor_labels_level6, anchor_boxes_level6 \
        = ds.get_next()

    # build optimizers
    global_step = tf.train.get_or_create_global_step()
    learning_rate = warmup_lr_schedule(init_learning_rate=cfg.TRAIN.BASE_LR, global_step=global_step,
                                       warmup_step=cfg.TRAIN.WARMUP_STEP)
    opt = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)

    sess_config = tf.ConfigProto()
    sess_config.allow_soft_placement = True
    sess_config.log_device_placement = False
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    if num_gpus > 1:

        base_inputs_list = [tf.split(value, num_or_size_splits=num_gpus, axis=0) for value in [
            images, gt_boxes, gt_labels, orig_gt_counts
        ]]
        fpn_all_anchors_list = \
            [[tf.identity(value) for _ in range(num_gpus)] for value in
             [all_anchors_level2, all_anchors_level3, all_anchors_level4, all_anchors_level5, all_anchors_level6]]
        fpn_anchor_gt_labels_list = \
            [tf.split(value, num_or_size_splits=num_gpus, axis=0) for value in
             [anchor_labels_level2, anchor_labels_level3, anchor_labels_level4,
              anchor_labels_level5, anchor_labels_level6]]
        fpn_anchor_gt_boxes_list = \
            [tf.split(value, num_or_size_splits=num_gpus, axis=0) for value in
             [anchor_boxes_level2, anchor_boxes_level3, anchor_boxes_level4,
              anchor_boxes_level5, anchor_boxes_level6]]

        tower_grads = []
        total_loss_dict = {'rpn_cls_loss': tf.constant(0.), 'rpn_box_loss': tf.constant(0.),
                           'rcnn_cls_loss': tf.constant(0.), 'rcnn_box_loss': tf.constant(0.)}
        for i, gpu_id in enumerate(gpus):
            with tf.device('/gpu:%d' % gpu_id):
                with tf.name_scope('model_%d' % gpu_id) as scope:
                    inputs1 = [input[i] for input in base_inputs_list]
                    inputs2 = [[input[i] for input in fpn_all_anchors_list]]
                    inputs3 = [[input[i] for input in fpn_anchor_gt_labels_list]]
                    inputs4 = [[input[i] for input in fpn_anchor_gt_boxes_list]]
                    net_inputs = inputs1 + inputs2 + inputs3 + inputs4
                    tower_loss_dict = tower_loss_func(net_inputs, reuse=(gpu_id > 0))
                    batch_norm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)

                    tower_loss = tf.add_n([v for k, v in tower_loss_dict.items()])

                    for k, v in tower_loss_dict.items():
                        total_loss_dict[k] += v

                    if i == num_gpus - 1:
                        wd_loss = regularize_cost('.*/kernel', l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), name='wd_cost')
                        tower_loss = tower_loss + wd_loss

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        if cfg.FRCNN.VISUALIZATION:
                            with tf.device('/cpu:0'):
                                with tf.name_scope('loss-summaries'):
                                    for k, v in tower_loss_dict.items():
                                        summaries.append(tf.summary.scalar(k, v))

                    grads = opt.compute_gradients(tower_loss)
                    tower_grads.append(grads)

        grads = average_gradients(tower_grads)
        for k, v in total_loss_dict.items():
            total_loss_dict[k] = v / tf.cast(num_gpus, tf.float32)
        average_total_loss = tf.add_n([v for k, v in total_loss_dict.items()] + [wd_loss])
    else:
        fpn_all_anchors = \
            [all_anchors_level2, all_anchors_level3, all_anchors_level4, all_anchors_level5, all_anchors_level6]
        fpn_anchor_gt_labels = \
            [anchor_labels_level2, anchor_labels_level3, anchor_labels_level4, anchor_labels_level5,
             anchor_labels_level6]
        fpn_anchor_gt_boxes = \
            [anchor_boxes_level2, anchor_boxes_level3, anchor_boxes_level4, anchor_boxes_level5, anchor_boxes_level6]
        net_inputs = [images, gt_boxes, gt_labels, orig_gt_counts,
                      fpn_all_anchors, fpn_anchor_gt_labels, fpn_anchor_gt_boxes]
        tower_loss_dict = tower_loss_func(net_inputs)
        batch_norm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        wd_loss = regularize_cost('.*/kernel', l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), name='wd_cost')
        average_total_loss = tf.add_n([v for k, v in tower_loss_dict.items()] + [wd_loss])
        grads = opt.compute_gradients(average_total_loss)
        total_loss_dict = tower_loss_dict

        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        if cfg.FRCNN.VISUALIZATION:
            with tf.device('/cpu:0'):
                with tf.name_scope('loss-summaries'):
                    for k, v in tower_loss_dict.items():
                        summaries.append(tf.summary.scalar(k, v))

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    summaries.append(tf.summary.scalar('learning_rate', learning_rate))

    # add histograms for trainable variables
    for grad, var in grads:
        # print(grad, var)
        if grad is not None:
            summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    # add histograms for trainable variables
    for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.op.name, var))

    variable_averages = tf.train.ExponentialMovingAverage(cfg.TRAIN.MOVING_AVERAGE_DECAY, num_updates=global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    all_global_vars = []
    for var in tf.global_variables():
        all_global_vars.append(var.name + '\n')
        # print(var.name, var.shape)
    with open('all_global_vars.txt', 'w') as fp:
        fp.writelines(all_global_vars)

    all_trainable_vars = []
    for var in tf.trainable_variables():
        all_trainable_vars.append(var.name + '\n')
    with open('all_trainable_vars.txt', 'w') as fp:
        fp.writelines(all_trainable_vars)

    all_moving_average_vars = []
    for var in tf.moving_average_variables():
        all_moving_average_vars.append(var.name + '\n')
    with open('all_moving_average_variables.txt', 'w') as fp:
        fp.writelines(all_moving_average_vars)

    # batch norm updates
    batch_norm_updates_op = tf.group(*batch_norm_updates)
    with tf.control_dependencies([apply_gradient_op, variable_averages_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge(summaries)
    summary_writer = tf.summary.FileWriter(checkpoint_path, tf.get_default_graph())

    init_op = tf.group([tf.global_variables_initializer(),
                        tf.local_variables_initializer()])
    sess.run(init_op)

    if False:
        print('load weights ...')
        ckpt_params = dict(np.load('MSRA-R50.npz'))
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

                if src_name == 'fc1000/W':
                    print('{} --> {} {}'.format('fc1000/W', dst_name, var.shape))
                    assign_ops.append(tf.assign(var, np.reshape(ckpt_params[src_name], [2048, 1000])))
                    continue
                if src_name in ckpt_params:
                    print('{} --> {} {}'.format(src_name, dst_name, var.shape))
                    assign_ops.append(tf.assign(var, ckpt_params[src_name]))
        print('load weights done.')
        with open('all_vars.txt', 'w') as fp:
            fp.writelines(all_variables)
        all_update_ops = []
        for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS):
            all_update_ops.append(op.name + '\n')
        with open('all_update_ops.txt', 'w') as fp:
            fp.writelines(all_update_ops)
        sess.run(assign_ops)
    else:
        if False:
            all_vars = []
            restore_var_dict = {}
            for var in tf.global_variables():
                all_vars.append(var.name+'\n')
                if 'rpn' not in var.name and 'rcnn' not in var.name and 'global_step' not in var.name and \
                        'Momentum' not in var.name and 'ExponentialMovingAverage' not in var.name:
                    restore_var_dict[var.name.replace(':0','')] = var
            with open('all_vars.txt', 'w') as fp:
                fp.writelines(all_vars)
            restorer = tf.train.Saver(var_list=restore_var_dict)
            restorer.restore(sess, cfg.BACKBONE.CHECKPOINT_PATH)
        else:
            if restore_from_original_checkpoint:
                # restore from official ResNet checkpoint
                all_vars = []
                restore_var_dict = {}
                for var in tf.global_variables():
                    all_vars.append(var.name + '\n')
                    if 'rpn' not in var.name and 'rcnn' not in var.name and 'fpn' not in var.name \
                            and 'global_step' not in var.name and \
                            'Momentum' not in var.name and 'ExponentialMovingAverage' not in var.name:
                        restore_var_dict[var.name.replace('resnet50/', '').replace(':0', '')] = var
                        print(var.name, var.shape)
                with open('all_vars.txt', 'w') as fp:
                    fp.writelines(all_vars)
                restore_vars_names = [k+'\n' for k in restore_var_dict.keys()]
                with open('all_restore_vars.txt', 'w') as fp:
                    fp.writelines(restore_vars_names)
                restorer = tf.train.Saver(var_list=restore_var_dict)
                restorer.restore(sess, cfg.BACKBONE.CHECKPOINT_PATH)
            else:
                all_vars = []
                restore_var_dict = {}
                for var in tf.global_variables():
                    all_vars.append(var.name + '\n')
                    restore_var_dict[var.name.replace(':0', '')] = var
                with open('all_vars.txt', 'w') as fp:
                    fp.writelines(all_vars)
                # restore from local checkpoint
                restorer = tf.train.Saver(tf.global_variables())
                try:
                    restorer.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
                except:
                    pass

    # record all ops
    all_operations = []
    for op in sess.graph.get_operations():
        all_operations.append(op.name + '\n')
    with open('all_ops.txt', 'w') as fp:
        fp.writelines(all_operations)

    loss_names = ['rpn_cls_loss', 'rpn_box_loss', 'rcnn_cls_loss', 'rcnn_box_loss']
    sess2run = list()
    sess2run.append(train_op)
    sess2run.append(learning_rate)
    sess2run.append(average_total_loss)
    sess2run.append(wd_loss)
    sess2run.extend([total_loss_dict[k] for k in loss_names])

    print('begin training ...')
    step = sess.run(global_step)
    step0 = step
    start = time.time()
    for step in range(step, cfg.TRAIN.MAX_STEPS):

        if step % cfg.TRAIN.SAVE_SUMMARY_STEPS == 0:

            _, lr_, tl_, wd_loss_, \
            rpn_cls_loss_, rpn_box_loss_, \
            rcnn_cls_loss_, rcnn_box_loss_, \
            summary_str = sess.run(sess2run + [summary_op])

            avg_time_per_step = (time.time() - start) / cfg.TRAIN.SAVE_SUMMARY_STEPS
            avg_examples_per_second = (cfg.TRAIN.SAVE_SUMMARY_STEPS * cfg.TRAIN.BATCH_SIZE_PER_GPU * num_gpus) \
                                      / (time.time() - start)
            start = time.time()
            print(
                'Step {:06d}, LR: {:.6f} LOSS: {:.4f}, '
                'RPN: {:.4f}, {:.4f}, RCNN: {:.4f}, {:.4f}, wd: {:.4f}, '
                '{:.2f} s/step, {:.2f} samples/s'.format(
                    step, lr_, tl_,
                    rpn_cls_loss_, rpn_box_loss_, rcnn_cls_loss_, rcnn_box_loss_, wd_loss_,
                    avg_time_per_step, avg_examples_per_second)
            )

            summary_writer.add_summary(summary_str, global_step=step)
        else:
            sess.run(train_op)

        if step % 1000 == 0:
            saver.save(sess, checkpoint_path + '/model.ckpt', global_step=step)

        # # profile the graph executation
        # if 1510 <= (step - step0) <= 1520:
        #     from tensorflow.python.client import timeline
        #     options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #     run_metadata = tf.RunMetadata()
        #     sess.run(train_op, options=options, run_metadata=run_metadata)
        #     fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        #     chrome_trace = fetched_timeline.generate_chrome_trace_format()
        #     with open('{}/timeline_step{}.json'.format(checkpoint_path, step), 'w') as fp:
        #         fp.write(chrome_trace)


def nms(dets, thresh):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap < thresh
    :return keep: indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def eval_one_dataset(dataset_name, output_filename):
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    import cv2
    from collections import namedtuple
    from dataset import DatasetRegistry
    from myaug_lib import short_side_resize_image
    DetectionResult = namedtuple('DetectionResult', ['box', 'score', 'class_id', 'mask'])
    register_coco(os.path.expanduser(cfg.DATA.BASEDIR))

    roidbs = DatasetRegistry.get(dataset_name).inference_roidbs()

    images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='images')
    with tf.variable_scope('resnet50'):
        final_boxes, final_scores, final_labels, final_inds = \
            model.model_fpn(images, is_training=False, data_format='channels_last', mode='test')

    init_op = tf.group([tf.global_variables_initializer(), tf.local_variables_initializer()])

    sess_config = tf.ConfigProto()
    sess_config.allow_soft_placement = True
    sess_config.log_device_placement = False
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    sess.run(init_op)

    checkpoint_path = cfg.TRAIN.LOG_DIR + COMMON_POSTFIX
    # restorer = tf.train.Saver()
    # restorer.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
    variable_averages = tf.train.ExponentialMovingAverage(decay=cfg.TRAIN.MOVING_AVERAGE_DECAY)
    variable_to_restore = variable_averages.variables_to_restore()
    restorer = tf.train.Saver(variable_to_restore)
    restorer.restore(sess, tf.train.latest_checkpoint(checkpoint_path))

    all_results = []
    start = time.time()
    for idx, roidb in enumerate(roidbs):
        fname, img_id = roidb["file_name"], roidb["image_id"]
        im = cv2.imread(fname, cv2.IMREAD_COLOR)
        im = im.astype("float32")
        h, w = im.shape[:2]

        # 短边resize
        resized_im = short_side_resize_image(im)
        # 减均值
        resized_im = resized_im[:, :, [2, 1, 0]]  # BGR-->RGB
        resized_im /= 255.0
        resized_im -= np.asarray(cfg.PREPROC.PIXEL_MEAN)
        resized_im /= np.asarray(cfg.PREPROC.PIXEL_STD)

        resized_h, resized_w = resized_im.shape[:2]

        scale = np.sqrt(resized_h * 1.0 / h * resized_w / w)

        mult = float(cfg.FPN.RESOLUTION_REQUIREMENT)  # size divisable
        max_height = int(np.ceil(float(resized_h) / mult) * mult)
        max_width = int(np.ceil(float(resized_w) / mult) * mult)
        resized_im1 = np.zeros((max_height, max_width, 3), dtype=np.float32)
        resized_im1[:resized_h, :resized_w, :] = resized_im

        # profile the graph executation
        if 1510 <= idx <= 1520:
            from tensorflow.python.client import timeline
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            boxes, scores, labels = sess.run([final_boxes, final_scores, final_labels],
                                             feed_dict={images: resized_im1[np.newaxis]},
                                             options=options, run_metadata=run_metadata)
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open('{}/timeline_Inference_step{}.json'.format(checkpoint_path, idx), 'w') as fp:
                fp.write(chrome_trace)
        else:
            boxes, scores, labels = sess.run([final_boxes, final_scores, final_labels],
                                             feed_dict={images: resized_im1[np.newaxis]})

        # Some slow numpy postprocessing:
        boxes = boxes / scale
        # boxes are already clipped inside the graph, but after the floating point scaling, this may not be true any more.
        boxes = boxes.reshape([-1, 4])
        boxes[:, [0, 1]] = np.maximum(boxes[:, [0, 1]], 0)
        boxes[:, 2] = np.minimum(boxes[:, 2], w-1)
        boxes[:, 3] = np.minimum(boxes[:, 3], h-1)

        if idx < 5:
            print(boxes, scores, labels)

        # if masks:
        #     full_masks = [_paste_mask(box, mask, orig_shape)
        #                   for box, mask in zip(boxes, masks[0])]
        #     masks = full_masks
        # else:
        #     # fill with none
        # masks = [None] * len(boxes)

        # postprocessing for FCOS
        # ################# 每一类进行nms ##################
        # boxes_after_nms = []
        # for c in range(1, 81):
        #     inds = np.where(labels == c)
        #     if len(inds) > 0:
        #         boxes_keep = np.concatenate([boxes[inds], scores[inds].reshape(-1, 1),
        #                                      labels[inds].reshape(-1, 1)], axis=1)
        #         # 类内NMS
        #         keep = nms(boxes_keep[:, 0:5], thresh=cfg.FCOS.NMS_THRESH)
        #         boxes_keep = boxes_keep[keep]
        #         # 过滤得分比较低的框
        #         # keep = np.where(boxes_keep[:, 4] > 0.1)  # 这里的阈值应该根据每一类来确定
        #         # boxes_keep = boxes_keep[keep]
        #         boxes_after_nms.append(boxes_keep)
        # boxes_after_nms = np.concatenate(boxes_after_nms, axis=0)  # [x1,y1,x2,y2,score,label]
        boxes_after_nms = np.concatenate([boxes, scores.reshape(-1, 1), labels.reshape(-1, 1)], axis=1)

        # ################# 限制每个图片最大检测个数 ##################
        number_of_detections = len(boxes_after_nms)
        if number_of_detections > cfg.FRCNN.TEST.RESULTS_PER_IM > 0:
            scores_sorted = np.sort(boxes_after_nms[:, 4])
            image_thresh = scores_sorted[number_of_detections - cfg.FRCNN.TEST.RESULTS_PER_IM + 1]
            keep = np.where(boxes_after_nms[:, 4] >= image_thresh)[0]
            boxes_after_nms = boxes_after_nms[keep]

        # ################# 类间nms ##################
        # keep = nms_across_class(boxes_after_nms, thresh=0.5)
        # boxes_after_nms = boxes_after_nms[keep]

        boxes = boxes_after_nms[:, 0:4]
        scores = boxes_after_nms[:, 4]
        labels = boxes_after_nms[:, 5].astype(np.int32)
        masks = [None] * len(boxes)

        for r in [DetectionResult(*args) for args in zip(boxes, scores, labels.tolist(), masks)]:
            res = {
                'image_id': img_id,
                'category_id': int(r.class_id),
                'bbox': [round(float(x), 4) for x in r.box],
                'score': round(float(r.score), 4),
            }
            all_results.append(res)

        if idx % 1000 == 0:
            print(idx, (time.time() - start) / 1000)
            start = time.time()

    DatasetRegistry.get(dataset_name).eval_inference_results(all_results, output_filename)


if __name__ == '__main__':

    action = sys.argv[1]

    if action == 'train':
        train()
    elif action == 'eval':

        for dataset_name in cfg.DATA.TEST:
            output_filename = '{}_predictions.json'.format(dataset_name)
            eval_one_dataset(dataset_name, output_filename)