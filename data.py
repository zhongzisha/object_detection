# -*- coding: utf-8 -*-
# @Time    : 2019/11/11 20:48
# @Author  : Zisha Zhong
# @Email   : zhongzisha@outlook.com
# @File    : data.py
# @Software: PyCharm

import os
import copy
import bisect
import itertools
from tabulate import tabulate
from termcolor import colored
import numpy as np
import cv2

import threading
from dataset import DatasetRegistry
import logger
from config import cfg
from tensorpack.dataflow import MultiProcessMapData, RepeatedData, RNGDataFlow, DataFromList, \
    MultiThreadMapData


if cfg.MODE_FRCNN or cfg.MODE_FPN:
    from fasterrcnn_model import get_all_anchors

if cfg.MODE_RETINANET:
    from retinanet_model import get_all_anchors_retinanet

from myaug_lib import random_horizontal_flip, random_vertical_flip, short_side_resize_image

try:
    import pycocotools.mask as cocomask

    # Much faster than utils/np_box_ops
    def np_iou(A, B):
        def to_xywh(box):
            box = box.copy()
            box[:, 2] -= box[:, 0]
            box[:, 3] -= box[:, 1]
            return box

        ret = cocomask.iou(
            to_xywh(A), to_xywh(B),
            np.zeros((len(B),), dtype=np.bool))
        # can accelerate even more, if using float32
        return ret.astype('float32')

except ImportError:
    from utils.np_box_ops import iou as np_iou  # noqa


class TrainingDataPreprocessor:
    """
    The mapper to preprocess the input data for training.

    Since the mapping may run in other processes, we write a new class and
    explicitly pass cfg to it, in the spirit of "explicitly pass resources to subprocess".
    """

    def __init__(self):

        self.noaug_set = set()
        if cfg.DATA.NOAUG_FILENAME is not None and os.path.exists(cfg.DATA.NOAUG_FILENAME):
            with open(cfg.DATA.NOAUG_FILENAME, 'r') as fp:
                self.noaug_set = set([f.strip() for f in fp.readlines()])

    def __call__(self, batch_roidbs):
        datapoint_list = []
        for roidb in batch_roidbs:
            fname, boxes, klass, is_crowd = roidb["file_name"], roidb["boxes"], roidb["class"], roidb["is_crowd"]
            boxes = np.copy(boxes)
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            im = im.astype("float32")

            # 随机水平翻转
            if np.random.randint(1, 2) == 1:
                im, boxes = random_horizontal_flip(im, boxes)

            # ##### augmentation #######
            # 短边resize
            im, boxes = short_side_resize_image(im, boxes)
            # 减均值
            im = im[:, :, [2, 1, 0]]   # BGR-->RGB
            im /= 255.0
            im -= np.asarray(cfg.PREPROC.PIXEL_MEAN)
            im /= np.asarray(cfg.PREPROC.PIXEL_STD)

            h, w = im.shape[:2]
            boxes = boxes.astype(np.int32)
            boxes[:, [0, 2]] = np.maximum(0, np.minimum(boxes[:, [0, 2]], w - 1))
            boxes[:, [1, 3]] = np.maximum(0, np.minimum(boxes[:, [1, 3]], h - 1))
            # ##### augmentation end ###

            boxes = boxes[is_crowd == 0]  # skip crowd boxes in training target
            klass = klass[is_crowd == 0]

            datapoint = {}
            datapoint["image"] = im
            datapoint["gt_boxes"] = boxes.astype(np.float32)
            datapoint["gt_labels"] = klass
            datapoint['filename'] = fname

            datapoint_list.append(datapoint)

        #################################################################################################################
        # Batchify the output
        #################################################################################################################

        batched_datapoint = {}
        # Require padding and original dimension storage
        # - image (HxWx3)
        # - gt_boxes (?x4)
        # - gt_labels (?)
        # - gt_masks (?xHxW)
        """
        Find the minimum container size for images (maxW x maxH)
        Find the maximum number of ground truth boxes
        For each image, save original dimension and pad
        """
        if cfg.PREPROC.PREDEFINED_PADDING:
            padding_shapes = [get_padding_shape(*(d["image"].shape[:2])) for d in datapoint_list]
            max_height = max([shp[0] for shp in padding_shapes])
            max_width = max([shp[1] for shp in padding_shapes])
        else:
            image_dims = [d["image"].shape for d in datapoint_list]
            heights = [dim[0] for dim in image_dims]
            widths = [dim[1] for dim in image_dims]

            max_height = max(heights)
            max_width = max(widths)

        mult = float(cfg.FPN.RESOLUTION_REQUIREMENT)  # size divisable
        max_height = int(np.ceil(float(max_height) / mult) * mult)
        max_width = int(np.ceil(float(max_width) / mult) * mult)
        # image
        padded_images = np.zeros((len(datapoint_list), max_height, max_width, 3), dtype=np.float32)
        original_image_dims = []
        for idx, datapoint in enumerate(datapoint_list):
            image = datapoint["image"]
            original_image_dims.append(image.shape)
            padded_images[idx, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        batched_datapoint["images"] = padded_images
        batched_datapoint["orig_image_dims"] = np.stack(original_image_dims)

        # gt_boxes and gt_labels
        max_num_gts = max([d["gt_labels"].size for d in datapoint_list])

        gt_counts = []
        padded_gt_labels = []
        padded_gt_boxes = []
        padded_gt_masks = []

        if cfg.MODE_FRCNN:
            anchor_labels = []
            anchor_boxes = []
            all_anchors = np.copy(get_all_anchors(
                max_height, max_width, stride=cfg.FRCNN.ANCHOR.STRIDE,
                sizes=cfg.FRCNN.ANCHOR.SIZES, ratios=cfg.FRCNN.ANCHOR.RATIOS
            ))

        if cfg.MODE_FPN:
            fpn_anchor_labels = []
            fpn_anchor_boxes = []
            fpn_all_anchors = []
            for lvl_idx, (stride, size) in enumerate(zip(cfg.FRCNN.FPN.ANCHOR_STRIDES, cfg.FRCNN.ANCHOR.SIZES)):
                fpn_all_anchors.append(np.copy(get_all_anchors(
                    max_height, max_width, stride=stride, sizes=(size,), ratios=cfg.FRCNN.ANCHOR.RATIOS
                )))
            fpn_all_anchors_flatten = [level_anchors.reshape((-1, 4)) for level_anchors in fpn_all_anchors]
            fpn_all_anchors_flatten = np.concatenate(fpn_all_anchors_flatten, axis=0)
            # filter the anchors outside the image
            fpn_anchors_indices_inside = np.where(
                (fpn_all_anchors_flatten[:, 0] >= 0) &
                (fpn_all_anchors_flatten[:, 1] >= 0) &
                (fpn_all_anchors_flatten[:, 2] < max_width) &
                (fpn_all_anchors_flatten[:, 3] < max_height))[0]
            fpn_all_inside_anchors = fpn_all_anchors_flatten[fpn_anchors_indices_inside, :]
            fpn_all_anchors_shapes = [anchors.shape for anchors in fpn_all_anchors]
            num_all_fpn_anchors = fpn_all_anchors_flatten.shape[0]

        if cfg.MODE_RETINANET:
            fpn_anchor_labels = []
            fpn_anchor_boxes = []
            fpn_all_anchors = []
            for lvl_idx, (stride, size) in enumerate(zip(cfg.RETINANET.ANCHOR_STRIDES, cfg.RETINANET.ANCHOR_SIZES)):
                fpn_all_anchors.append(np.copy(get_all_anchors_retinanet(
                    max_height, max_width, stride=stride, sizes=(size,), ratios=cfg.RETINANET.ANCHOR_RATIOS
                )))
            fpn_all_anchors_flatten = [level_anchors.reshape((-1, 4)) for level_anchors in fpn_all_anchors]
            fpn_all_anchors_flatten = np.concatenate(fpn_all_anchors_flatten, axis=0)
            # filter the anchors outside the image
            fpn_anchors_indices_inside = np.where(
                (fpn_all_anchors_flatten[:, 0] >= 0) &
                (fpn_all_anchors_flatten[:, 1] >= 0) &
                (fpn_all_anchors_flatten[:, 2] < max_width) &
                (fpn_all_anchors_flatten[:, 3] < max_height))[0]
            fpn_all_inside_anchors = fpn_all_anchors_flatten[fpn_anchors_indices_inside, :]
            fpn_all_anchors_shapes = [anchors.shape for anchors in fpn_all_anchors]
            num_all_fpn_anchors = fpn_all_anchors_flatten.shape[0]

        for idx, datapoint in enumerate(datapoint_list):
            gt_count_for_image = datapoint["gt_labels"].size
            gt_counts.append(gt_count_for_image)

            gt_padding = max_num_gts - gt_count_for_image

            padded_gt_labels_for_img = np.pad(datapoint["gt_labels"], [0, gt_padding], 'constant',
                                              constant_values=-1)
            padded_gt_labels.append(padded_gt_labels_for_img)

            padded_gt_boxes_for_img = np.pad(datapoint["gt_boxes"],
                                             [[0, gt_padding],
                                              [0, 0]],
                                             'constant')
            padded_gt_boxes.append(padded_gt_boxes_for_img)

            # h_padding = max_height - datapoint["image"].shape[0]
            # w_padding = max_width - datapoint["image"].shape[1]

            # if cfg.MODE_MASK:
            #     padded_gt_masks_for_img = np.pad(datapoint["gt_masks"],
            #                                      [[0, gt_padding],
            #                                       [0, h_padding],
            #                                       [0, w_padding]],
            #                                      'constant')
            #     padded_gt_masks.append(padded_gt_masks_for_img)

            if cfg.MODE_FRCNN:
                anchor_labels_current, anchor_boxes_current = self.get_rpn_anchor_input(
                    padded_images[idx], datapoint["gt_boxes"], stride=cfg.FRCNN.ANCHOR.STRIDE
                )
                anchor_labels.append(anchor_labels_current)
                anchor_boxes.append(anchor_boxes_current)

            if cfg.MODE_FPN:
                # for FPN anchor inputs
                # anchor_labels_current, anchor_boxes_current = self.get_multilevel_rpn_anchor_input(
                #     padded_images[idx], datapoint["gt_boxes"],
                #     strides=cfg.FRCNN.FPN.ANCHOR_STRIDES, sizes=cfg.FRCNN.ANCHOR.SIZES,
                #     ratios=cfg.FRCNN.ANCHOR.RATIOS
                # )
                anchor_labels_current, anchor_boxes_current = self.get_multilevel_rpn_anchor_input_simplied(
                    datapoint["gt_boxes"],
                    fpn_all_inside_anchors, fpn_anchors_indices_inside,
                    num_all_fpn_anchors, fpn_all_anchors_shapes,
                    strides=cfg.FRCNN.FPN.ANCHOR_STRIDES,
                    sizes=cfg.FRCNN.ANCHOR.SIZES,
                    ratios=cfg.FRCNN.ANCHOR.RATIOS)
                fpn_anchor_labels.append(anchor_labels_current)
                fpn_anchor_boxes.append(anchor_boxes_current)

            if cfg.MODE_RETINANET:
                anchor_labels_current, anchor_boxes_current = self.get_retinanet_anchor_input_simplied(
                    datapoint["gt_boxes"],
                    datapoint["gt_labels"],
                    fpn_all_inside_anchors, fpn_anchors_indices_inside,
                    num_all_fpn_anchors, fpn_all_anchors_shapes,
                    strides=cfg.RETINANET.ANCHOR_STRIDES,
                    sizes=cfg.RETINANET.ANCHOR_SIZES,
                    ratios=cfg.RETINANET.ANCHOR_RATIOS)
                # print(anchor_labels_current[0].shape, anchor_boxes_current[0].shape)
                fpn_anchor_labels.append(anchor_labels_current)
                fpn_anchor_boxes.append(anchor_boxes_current)

        # basic inputs
        batched_datapoint["orig_gt_counts"] = np.stack(gt_counts)
        batched_datapoint["gt_labels"] = np.stack(padded_gt_labels)
        batched_datapoint["gt_boxes"] = np.stack(padded_gt_boxes)
        batched_datapoint["filenames"] = [d["filename"] for d in datapoint_list]

        if cfg.MODE_FRCNN:
            batched_datapoint["all_anchors"] = all_anchors
            batched_datapoint["anchor_labels"] = np.stack(anchor_labels)
            batched_datapoint["anchor_boxes"] = np.stack(anchor_boxes)

        if cfg.MODE_FPN:
            for idx, stride in enumerate(cfg.FRCNN.FPN.ANCHOR_STRIDES):
                batched_datapoint["all_anchors_level{}".format(2+idx)] = fpn_all_anchors[idx]
                batched_datapoint["anchor_labels_level{}".format(2+idx)] = \
                    np.stack([fpn_anchor_labels[im_i][idx] for im_i in range(len(datapoint_list))])
                batched_datapoint["anchor_boxes_level{}".format(2+idx)] = \
                    np.stack([fpn_anchor_boxes[im_i][idx] for im_i in range(len(datapoint_list))])

        if cfg.MODE_RETINANET:
            # p3, p4, p5, p6, p7
            for idx, stride in enumerate(cfg.FRCNN.FPN.ANCHOR_STRIDES):
                batched_datapoint["all_anchors_level{}".format(3+idx)] = fpn_all_anchors[idx]
                batched_datapoint["anchor_labels_level{}".format(3+idx)] = \
                    np.stack([fpn_anchor_labels[im_i][idx] for im_i in range(len(datapoint_list))])
                batched_datapoint["anchor_boxes_level{}".format(3+idx)] = \
                    np.stack([fpn_anchor_boxes[im_i][idx] for im_i in range(len(datapoint_list))])

        # if cfg.MODE_MASK:
        #     batched_datapoint["gt_masks"] = np.stack(padded_gt_masks)

        return batched_datapoint

    def get_anchor_labels(self, anchors, gt_boxes):
        '''
        label each anchor as fg/bg/ignore
        :param anchors: Ax4
        :param gt_boxes: Bx4
        :return:
            anchor_labels: (A,), int, each element is {-1, 0, 1}
            anchor_boxes: Ax4, the target gt_box for each anchor when the anchor is fg
        '''

        def filter_box_label(labels, value, max_num):
            curr_inds = np.where(labels == value)[0]
            if len(curr_inds) > max_num:
                disable_inds = np.random.choice(curr_inds, size=(len(curr_inds) - max_num), replace=False)
                labels[disable_inds] = -1
                curr_inds = np.where(labels == value)[0]
            return curr_inds

        NA, NB = len(anchors), len(gt_boxes)
        if NB == 0:
            anchor_labels = np.zeros((NA,), dtype=np.int32)
            filter_box_label(anchor_labels, 0, max_num=cfg.FRCNN.RPN.BATCH_PER_IM)
            return anchor_labels, np.zeros((NA, 4), dtype=np.float32)

        box_ious = np_iou(anchors, gt_boxes)  # NAxNB
        ious_argmax_per_anchor = box_ious.argmax(axis=1)  # NA
        ious_max_per_anchor = box_ious.max(axis=1)
        ious_max_per_gt = np.amax(box_ious, axis=0, keepdims=True) # 1xNB
        anchors_with_max_iou_per_gt = np.where(box_ious == ious_max_per_gt)[0]

        anchor_labels = -np.ones((NA, ), dtype=np.int32)
        anchor_labels[anchors_with_max_iou_per_gt] = 1
        anchor_labels[ious_max_per_anchor >= cfg.FRCNN.RPN.POSITIVE_ANCHOR_THRESH] = 1
        anchor_labels[ious_max_per_anchor < cfg.FRCNN.RPN.NEGATIVE_ANCHOR_THRESH] = 0

        # label all non-ignore candidate boxes which overlap crowd as ignore
        # if crowd_boxes.size > 0:
        #     cand_inds = np.where(anchor_labels >= 0)[0]
        #     cand_anchors = anchors[cand_inds]
        #     ioas = np_ioa(crowd_boxes, cand_anchors)
        #     overlap_with_crowd = cand_inds[ioas.max(axis=0) > cfg.FRCNN.RPN.CROWD_OVERLAP_THRESH]
        #     anchor_labels[overlap_with_crowd] = -1

        # subsample fg labels: ignore some fg if fg is too many
        target_sum_fg = int(cfg.FRCNN.RPN.BATCH_PER_IM * cfg.FRCNN.RPN.FG_RATIO)
        fg_inds = filter_box_label(anchor_labels, 1, target_sum_fg)

        # subsample bg labels: num_bg is not allowed to be too many
        old_num_bg = np.sum(anchor_labels == 0)
        if old_num_bg == 0:
            raise BaseException
        target_num_bg = cfg.FRCNN.RPN.BATCH_PER_IM - len(fg_inds)
        filter_box_label(anchor_labels, 0, target_num_bg)

        anchor_boxes = np.zeros((NA, 4), dtype=np.float32)
        fg_boxes = gt_boxes[ious_argmax_per_anchor[fg_inds], :]
        anchor_boxes[fg_inds, :] = fg_boxes

        return anchor_labels, anchor_boxes

    def get_rpn_anchor_input(self, im, gt_boxes, stride, sizes=cfg.FRCNN.ANCHOR.SIZES):
        boxes = np.copy(gt_boxes)
        height, width = im.shape[:2]
        all_anchors = np.copy(get_all_anchors(
            height, width, stride=stride,
            sizes=sizes,
            ratios=cfg.FRCNN.ANCHOR.RATIOS
        ))
        # fHxfWx4 --> (-1, 4)
        anchors_flatten = all_anchors.reshape((-1, 4))
        # filter the anchors outside the image
        indices_inside = np.where(
            (anchors_flatten[:, 0] >= 0) &
            (anchors_flatten[:, 1] >= 0) &
            (anchors_flatten[:, 2] < width) &
            (anchors_flatten[:, 3] < height))[0]
        inside_anchors = anchors_flatten[indices_inside, :]

        # get anchor labels and their corresponding gt boxes
        anchor_labels, anchor_gt_boxes = self.get_anchor_labels(inside_anchors, boxes)

        anchorH, anchorW = all_anchors.shape[:2]
        num_anchor = len(sizes) * len(cfg.FRCNN.ANCHOR.RATIOS)
        featuremap_labels = -np.ones((anchorH * anchorW * num_anchor, ), dtype=np.int32)
        featuremap_labels[indices_inside] = anchor_labels
        featuremap_labels = featuremap_labels.reshape((anchorH, anchorW, num_anchor))
        featuremap_boxes = np.zeros((anchorH * anchorW * num_anchor, 4), dtype=np.float32)
        featuremap_boxes[indices_inside, :] = anchor_gt_boxes
        featuremap_boxes = featuremap_boxes.reshape((anchorH, anchorW, num_anchor, 4))

        return featuremap_labels, featuremap_boxes

    def get_multilevel_rpn_anchor_input(self, im, gt_boxes, strides, sizes, ratios):
        boxes = np.copy(gt_boxes)
        height, width = im.shape[:2]
        all_anchors = []
        for lvl_idx, (stride, size) in enumerate(zip(strides, sizes)):
            all_anchors.append(np.copy(get_all_anchors(
                height, width, stride=stride,
                sizes=(size,),
                ratios=ratios
            )))
        all_anchors_flatten = [level_anchors.reshape((-1, 4)) for level_anchors in all_anchors]
        all_anchors_flatten = np.concatenate(all_anchors_flatten, axis=0)
        # filter the anchors outside the image
        indices_inside = np.where(
            (all_anchors_flatten[:, 0] >= 0) &
            (all_anchors_flatten[:, 1] >= 0) &
            (all_anchors_flatten[:, 2] < width) &
            (all_anchors_flatten[:, 3] < height))[0]
        inside_anchors = all_anchors_flatten[indices_inside, :]

        # get anchor labels and their corresponding gt boxes
        anchor_labels, anchor_gt_boxes = self.get_anchor_labels(inside_anchors, boxes)

        num_all_anchors = all_anchors_flatten.shape[0]
        all_labels = -np.ones((num_all_anchors,), dtype=np.int32)
        all_labels[indices_inside] = anchor_labels
        all_boxes = np.zeros((num_all_anchors, 4), dtype=np.float32)
        all_boxes[indices_inside, :] = anchor_gt_boxes

        featuremap_labels = []
        featuremap_boxes = []
        start = 0
        for lvl_idx, (stride, size) in enumerate(zip(strides, sizes)):
            anchorH, anchorW, num_anchor = all_anchors[lvl_idx].shape[:3]
            level_length = anchorH * anchorW * num_anchor
            end = start + level_length
            featuremap_labels.append(all_labels[start:end].reshape((anchorH, anchorW, num_anchor)))
            featuremap_boxes.append(all_boxes[start:end, :].reshape((anchorH, anchorW, num_anchor, 4)))
            start = end

        return featuremap_labels, featuremap_boxes

    def get_multilevel_rpn_anchor_input_simplied(self, gt_boxes, inside_anchors, indices_inside, num_all_anchors,
                                                 all_anchors_shapes, strides, sizes, ratios):
        boxes = np.copy(gt_boxes)

        # get anchor labels and their corresponding gt boxes
        anchor_labels, anchor_gt_boxes = self.get_anchor_labels(inside_anchors, boxes)

        all_labels = -np.ones((num_all_anchors,), dtype=np.int32)
        all_labels[indices_inside] = anchor_labels
        all_boxes = np.zeros((num_all_anchors, 4), dtype=np.float32)
        all_boxes[indices_inside, :] = anchor_gt_boxes

        featuremap_labels = []
        featuremap_boxes = []
        start = 0
        for lvl_idx, (stride, size) in enumerate(zip(strides, sizes)):
            anchorH, anchorW, num_anchor = all_anchors_shapes[lvl_idx][:3]
            level_length = anchorH * anchorW * num_anchor
            end = start + level_length
            featuremap_labels.append(all_labels[start:end].reshape((anchorH, anchorW, num_anchor)))
            featuremap_boxes.append(all_boxes[start:end, :].reshape((anchorH, anchorW, num_anchor, 4)))
            start = end

        return featuremap_labels, featuremap_boxes

    # for RetinaNet
    def get_retinanet_anchor_labels(self, anchors, gt_boxes, gt_labels):
        '''
        label each anchor as fg/bg/ignore
        :param anchors: Ax4
        :param gt_boxes: Bx4
        :param gt_labels: Bx1
        :return:
            anchor_labels: (A,), int, each element is {-1, 0, C}  C is in [1, NUM_CATEGORY+1]
            anchor_boxes: Ax4, the target gt_box for each anchor when the anchor is fg
        '''

        NA, NB = len(anchors), len(gt_boxes)
        if NB == 0:
            anchor_labels = np.zeros((NA,), dtype=np.int32)
            return anchor_labels, np.zeros((NA, 4), dtype=np.float32)

        box_ious = np_iou(anchors, gt_boxes)  # NAxNB
        ious_argmax_per_anchor = box_ious.argmax(axis=1)  # NA
        ious_max_per_anchor = box_ious.max(axis=1)
        ious_max_per_gt = np.amax(box_ious, axis=0, keepdims=True) # 1xNB
        anchors_with_max_iou_per_gt = np.where(box_ious == ious_max_per_gt)[0]

        anchor_labels = -np.ones((NA, ), dtype=np.int32)
        anchor_labels[anchors_with_max_iou_per_gt] = 1
        anchor_labels[ious_max_per_anchor >= cfg.RETINANET.POSITIVE_ANCHOR_THRESH] = 1
        anchor_labels[ious_max_per_anchor < cfg.RETINANET.NEGATIVE_ANCHOR_THRESH] = 0

        # label all non-ignore candidate boxes which overlap crowd as ignore
        # if crowd_boxes.size > 0:
        #     cand_inds = np.where(anchor_labels >= 0)[0]
        #     cand_anchors = anchors[cand_inds]
        #     ioas = np_ioa(crowd_boxes, cand_anchors)
        #     overlap_with_crowd = cand_inds[ioas.max(axis=0) > cfg.FRCNN.RPN.CROWD_OVERLAP_THRESH]
        #     anchor_labels[overlap_with_crowd] = -1

        fg_inds = np.where(anchor_labels == 1)[0]
        anchor_boxes = np.zeros((NA, 4), dtype=np.float32)
        fg_boxes = gt_boxes[ious_argmax_per_anchor[fg_inds], :]
        fg_labels = gt_labels[ious_argmax_per_anchor[fg_inds]].reshape(-1)
        anchor_boxes[fg_inds, :] = fg_boxes
        anchor_labels[fg_inds] = fg_labels   # [0, 80]

        return anchor_labels, anchor_boxes

    # for RetinaNet
    def get_retinanet_anchor_input_simplied(self, gt_boxes, gt_labels,
                                            inside_anchors, indices_inside, num_all_anchors,
                                            all_anchors_shapes, strides, sizes, ratios):
        boxes = np.copy(gt_boxes)
        labels = np.copy(gt_labels).reshape(-1, 1)

        # get anchor labels and their corresponding gt boxes
        anchor_labels, anchor_gt_boxes = self.get_retinanet_anchor_labels(inside_anchors, boxes, labels)

        all_labels = -np.ones((num_all_anchors,), dtype=np.int32)
        all_labels[indices_inside] = anchor_labels
        all_boxes = np.zeros((num_all_anchors, 4), dtype=np.float32)
        all_boxes[indices_inside, :] = anchor_gt_boxes

        featuremap_labels = []
        featuremap_boxes = []
        start = 0
        for lvl_idx, (stride, size) in enumerate(zip(strides, sizes)):
            anchorH, anchorW, num_anchor = all_anchors_shapes[lvl_idx][:3]
            level_length = anchorH * anchorW * num_anchor
            end = start + level_length
            featuremap_labels.append(all_labels[start:end].reshape((anchorH, anchorW, num_anchor)))
            featuremap_boxes.append(all_boxes[start:end, :].reshape((anchorH, anchorW, num_anchor, 4)))
            start = end

        return featuremap_labels, featuremap_boxes


def print_class_histogram(roidbs):
    """
    Args:
        roidbs (list[dict]): the same format as the output of `training_roidbs`.
    """
    class_names = DatasetRegistry.get_metadata(cfg.DATA.TRAIN[0], 'class_names')
    # labels are in [1, NUM_CATEGORY], hence +2 for bins
    hist_bins = np.arange(cfg.DATA.NUM_CATEGORY + 2)

    # Histogram of ground-truth objects
    gt_hist = np.zeros((cfg.DATA.NUM_CATEGORY + 1,), dtype=np.int)
    for entry in roidbs:
        # filter crowd?
        gt_inds = np.where((entry["class"] > 0) & (entry["is_crowd"] == 0))[0]
        gt_classes = entry["class"][gt_inds]
        if len(gt_classes):
            assert gt_classes.max() <= len(class_names) - 1
        gt_hist += np.histogram(gt_classes, bins=hist_bins)[0]
    data = list(itertools.chain(*[[class_names[i + 1], v] for i, v in enumerate(gt_hist[1:])]))
    COL = min(6, len(data))
    total_instances = sum(data[1::2])
    data.extend([None] * ((COL - len(data) % COL) % COL))
    data.extend(["total", total_instances])
    data = itertools.zip_longest(*[data[i::COL] for i in range(COL)])
    # the first line is BG
    table = tabulate(data, headers=["class", "#box"] * (COL // 2), tablefmt="pipe", stralign="center", numalign="left")
    logger.info("Ground-Truth category distribution:\n" + colored(table, "cyan"))


def _get_padding_shape(aspect_ratio):
    for shape in cfg.PREPROC.PADDING_SHAPES:
        if aspect_ratio >= float(shape[0])/float(shape[1]):
            return shape

    return cfg.PREPROC.PADDING_SHAPES[-1]


def get_padding_shape(h, w):
    aspect_ratio = float(h)/float(w)

    if aspect_ratio > 1.0:
        inv = 1./aspect_ratio
    else:
        inv = aspect_ratio

    shp = _get_padding_shape(inv)
    if aspect_ratio > 1.0:
        return (shp[1], shp[0])
    return shp


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


class AspectGroupingDataFlow(RNGDataFlow):
    '''
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/samplers/grouped_batch_sampler.py
    '''

    def __init__(self, roidbs, group_ids, batch_size=2,  drop_uneven=True):
        super(AspectGroupingDataFlow, self).__init__()
        self.roidbs = roidbs
        self.group_ids = group_ids
        self.batch_size = batch_size
        self.drop_uneven = drop_uneven

        self.groups = np.sort(np.unique(self.group_ids))
        self._can_reuse_batches = False

    def __iter__(self):
        if self._can_reuse_batches:
            batches = self._batches
            self._can_reuse_batches = False
        else:
            batches = self._prepare_batches()
        self._batches = batches
        for ids in iter(batches):
            batch_roidbs = [self.roidbs[idx] for idx in ids]
            yield batch_roidbs

    def get_data(self):
        return self.__iter__()

    def __len__(self):
        if not hasattr(self, '_batches'):
            self._batches = self._prepare_batches()
            self._can_reuse_batches = True
        return len(self._batches)

    def _prepare_batches(self):
        dataset_size = len(self.group_ids)
        sampled_ids = np.random.permutation(len(self.roidbs))
        order = np.full((dataset_size,), fill_value=-1, dtype=np.int64)
        order[sampled_ids] = np.arange(len(sampled_ids))
        mask = order >= 0
        clusters = [(self.group_ids == i) & mask for i in self.groups]
        relative_order = [order[cluster] for cluster in clusters]
        permutation_ids = [np.sort(s) for s in relative_order]
        permuted_clusters = [sampled_ids[idx] for idx in permutation_ids]
        # splits = [np.array_split(c, int(np.ceil(len(c)/self.batch_size))) for c in permuted_clusters]
        splits = [np.split(c, self.batch_size * np.arange(1, int(np.ceil(len(c) / self.batch_size))))
                  for c in permuted_clusters]
        merged = tuple(itertools.chain.from_iterable(splits))
        first_element_of_batch = [t[0] for t in merged]
        inv_sampled_ids_map = {v: k for k,v in enumerate(sampled_ids.tolist())}
        first_index_of_batch = [inv_sampled_ids_map[s] for s in first_element_of_batch]
        permutation_order = np.argsort(first_index_of_batch)
        batches = [merged[i].tolist() for i in permutation_order]

        if self.drop_uneven:
            kept = []
            for batch in batches:
                if len(batch) == self.batch_size:
                    kept.append(batch)
                batches = kept
        return batches


def get_train_dataflow(batch_size=2):
    print("In train dataflow")
    roidbs = list(itertools.chain.from_iterable(DatasetRegistry.get(x).training_roidbs() for x in cfg.DATA.TRAIN))
    print_class_histogram(roidbs)
    print("Done loading roidbs")

    # Filter out images that have no gt boxes, but this filter shall not be applied for testing.
    # The model does support training with empty images, but it is not useful for COCO.
    num = len(roidbs)
    roidbs = list(filter(lambda img: len(img["boxes"][img["is_crowd"] == 0]) > 0, roidbs))
    logger.info(
        "Filtered {} images which contain no non-crowd groudtruth boxes. Total #images for training: {}".format(
            num - len(roidbs), len(roidbs)
        )
    )

    aspect_grouping = [1]
    aspect_ratios = [float(x["height"]) / float(x["width"]) for x in roidbs]
    group_ids = _quantize(aspect_ratios, aspect_grouping)

    ds = AspectGroupingDataFlow(roidbs, group_ids, batch_size=batch_size, drop_uneven=True)
    preprocess = TrainingDataPreprocessor()
    buffer_size = cfg.DATA.NUM_WORKERS * 10
    # ds = MultiProcessMapData(ds, cfg.DATA.NUM_WORKERS, preprocess, buffer_size=buffer_size)
    ds = MultiThreadMapData(ds, cfg.DATA.NUM_WORKERS, preprocess, buffer_size=buffer_size)
    ds.reset_state()

    # to get an infinite data flow
    ds = RepeatedData(ds, num=-1)
    dataiter = ds.__iter__()

    return dataiter


def get_plain_train_dataflow(batch_size=2):
    # no aspect ratio grouping

    print("In train dataflow")
    roidbs = list(itertools.chain.from_iterable(DatasetRegistry.get(x).training_roidbs() for x in cfg.DATA.TRAIN))
    print_class_histogram(roidbs)
    print("Done loading roidbs")

    # Filter out images that have no gt boxes, but this filter shall not be applied for testing.
    # The model does support training with empty images, but it is not useful for COCO.
    num = len(roidbs)
    roidbs = list(filter(lambda img: len(img["boxes"][img["is_crowd"] == 0]) > 0, roidbs))
    logger.info(
        "Filtered {} images which contain no non-crowd groudtruth boxes. Total #images for training: {}".format(
            num - len(roidbs), len(roidbs)
        )
    )

    ds = DataFromList(roidbs, shuffle=True)
    preprocess = TrainingDataPreprocessor()
    buffer_size = cfg.DATA.NUM_WORKERS * 20
    ds = MultiProcessMapData(ds, cfg.DATA.NUM_WORKERS, preprocess, buffer_size=buffer_size)
    ds.reset_state()
    dataiter = ds.__iter__()
    return dataiter


class PrefetchingIter:

    def __init__(self, iters, num_gpu):
        self.iters = iters
        self.n_iter = 1
        self.data_ready = [threading.Event() for _ in range(self.n_iter)]
        self.data_taken = [threading.Event() for _ in range(self.n_iter)]
        for e in self.data_taken:
            e.set()
        self.started = True
        self.current_batch = [None for _ in range(self.n_iter)]
        self.next_batch = [None for _ in range(self.n_iter)]

        def prefetch_func(self, i):
            while True:
                self.data_taken[i].wait()
                if not self.started:
                    break
                try:
                    blobs_list = []
                    cnt = 0
                    while cnt < num_gpu:
                        blobs = next(iters)
                        blobs_list.append(blobs)
                        cnt += 1
                    self.next_batch[i] = blobs_list
                except StopIteration:
                    self.next_batch[i] = None
                self.data_taken[i].clear()
                self.data_ready[i].set()

        self.prefetch_threads = [
            threading.Thread(target=prefetch_func, args=[self, i])
            for i in range(self.n_iter)
        ]
        for thread in self.prefetch_threads:
            thread.setDaemon(True)
            thread.start()

    def __del__(self):
        self.started = False
        for e in self.data_taken:
            e.set()
        for thread in self.prefetch_threads:
            thread.join()

    def iter_next(self):
        for e in self.data_ready:
            e.wait()
        if self.next_batch[0] is None:
            return False
        else:
            self.current_batch = self.next_batch[0]
            for e in self.data_ready:
                e.clear()
            for e in self.data_taken:
                e.set()
            return True

    def forward(self):
        if self.iter_next():
            return self.current_batch
        else:
            return StopIteration


def get_pascal_voc_train_dataflow(batch_size=1):
    from dataset import register_pascal_voc

    # register_coco(os.path.expanduser("/media/ubuntu/Working/common_data/coco"))
    register_pascal_voc(os.path.expanduser("/media/ubuntu/Working/voc2012/VOC2012/"))

    print("In train dataflow")
    roidbs = list(itertools.chain.from_iterable(DatasetRegistry.get(x).training_roidbs() for x in cfg.DATA.TRAIN))
    print_class_histogram(roidbs)
    print("Done loading roidbs")

    # Filter out images that have no gt boxes, but this filter shall not be applied for testing.
    # The model does support training with empty images, but it is not useful for COCO.
    num = len(roidbs)
    roidbs = list(filter(lambda img: len(img["boxes"][img["is_crowd"] == 0]) > 0, roidbs))
    logger.info(
        "Filtered {} images which contain no non-crowd groudtruth boxes. Total #images for training: {}".format(
            num - len(roidbs), len(roidbs)
        )
    )

    aspect_grouping = [1]
    aspect_ratios = [float(x["height"]) / float(x["width"]) for x in roidbs]
    group_ids = _quantize(aspect_ratios, aspect_grouping)

    ds = DataFromList(np.arange(len(roidbs)), shuffle=True)
    ds.reset_state()
    ds = AspectGroupingDataFlow(roidbs, ds, group_ids, batch_size=batch_size, drop_uneven=True).__iter__()
    preprocess = TrainingDataPreprocessor()

    while True:
        batch_roidbs = next(ds)
        yield preprocess(batch_roidbs)


def using_tf_dataset():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import time
    import tensorflow as tf
    from dataset import register_coco

    num_gpus = 2

    register_coco(os.path.expanduser("/media/ubuntu/Working/common_data/coco"))
    # register_pascal_voc(os.path.expanduser("/media/ubuntu/Working/voc2012/VOC2012/"))

    data_iter = get_train_dataflow(batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU)
    ds = tf.data.Dataset()
    ds = ds.from_generator(lambda: map(lambda x: tuple([x[k] for k in
                                                        ['images', 'gt_boxes', 'gt_labels', 'orig_gt_counts',
                                                         'anchor_labels', 'anchor_boxes']]),
                                       data_iter),
                           (tf.float32, tf.float32, tf.int64, tf.int32, tf.int32, tf.float32),
                           (tf.TensorShape([None, None, None, 3]),
                            tf.TensorShape([None, None, 4]),
                            tf.TensorShape([None, None]),
                            tf.TensorShape([None, ]),
                            tf.TensorShape([None, None, None, None]),
                            tf.TensorShape([None, None, None, None, 4])))
    # # ds = ds.map(identity_func, num_parallel_calls=1)
    # ds = ds.interleave(identity_func, cycle_length=3, block_length=1, num_parallel_calls=3)
    ds = ds.prefetch(buffer_size=512)
    ds = ds.make_one_shot_iterator()
    net_inputs = ds.get_next()
    images, gt_boxes, gt_labels, orig_gt_counts, anchor_labels, anchor_boxes = net_inputs

    sess_config = tf.ConfigProto()
    sess_config.allow_soft_placement = True
    sess_config.log_device_placement = False
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    # inputs_list = []
    # for i in range(num_gpus):
    #     inputs_list.append(net_inputs)
    # put_op_list = []
    # get_op_list = []
    # for i in range(num_gpus):
    #     with tf.device("/GPU:{}".format(i)):
    #         area = StagingArea(dtypes=[tf.float32, tf.float32, tf.int64, tf.int32, tf.int32, tf.float32],
    #                            shapes=(tf.TensorShape([None, None, None, 3]),
    #                                    tf.TensorShape([None, None, 4]),
    #                                    tf.TensorShape([None, None]),
    #                                    tf.TensorShape([None, ]),
    #                                    tf.TensorShape([None, None, None, None]),
    #                                    tf.TensorShape([None, None, None, None, 4])
    #                                    ),
    #                            capacity=64)
    #         put_op_list.append(area.put(inputs_list[i]))
    #         get_op_list.append(area.get())

    start = time.time()
    for step in range(10000000000000000):

        images_, gt_boxes_, gt_labels_, orig_gt_counts_, anchor_labels_, anchor_boxes_ = \
            sess.run([images, gt_boxes, gt_labels, orig_gt_counts, anchor_labels, anchor_boxes])

        if step %1000 == 0:

            print(step, (time.time() - start) / 1000)
            start = time.time()

        # images_ += np.asarray(cfg.PREPROC.PIXEL_MEAN)
        #
        # img1 = images_[0].astype(np.uint8)
        # img1 = draw_boxes(img1, gt_boxes_[0], gt_labels_[0])
        # cv2.imwrite('img.png', img1)
        #
        # anchor_labels_ = anchor_labels_[0].reshape(-1,)
        # anchor_boxes_ = anchor_boxes_[0].reshape(-1, 4)
        # pos_indices = np.where(anchor_labels_ == 1)[0]
        # print(pos_indices)
        # pos_boxes = anchor_boxes_[pos_indices, :]
        # pos_labels = anchor_labels_[pos_indices]
        #
        # print(gt_boxes_[0])
        # print(pos_boxes)
        #
        # img2 = draw_boxes(img1, pos_boxes, pos_labels, color=(0, 0, 255))
        # cv2.imwrite('img2.png', img2)

        # data_sample = dict(images=images_, gt_boxes=gt_boxes_, gt_labels=gt_labels_,
        #                    orig_gt_counts=orig_gt_counts_,
        #                    anchor_labels=anchor_labels_, anchor_boxes=anchor_boxes_)
        # with open('data_sample.npz', 'wb') as fp:
        #     np.savez(fp, **data_sample)


def test_pl():
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import time
    import tensorflow as tf
    from dataset import register_coco

    num_gpus = 2

    register_coco(os.path.expanduser("/media/ubuntu/Working/coco"))
    # register_pascal_voc(os.path.expanduser("/media/ubuntu/Working/voc2012/VOC2012/"))

    print('get dataflow ...')
    data_iter = get_train_dataflow(batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU)

    images = tf.placeholder(tf.float32, shape=[None, None, None, 3])  # images
    gt_boxes = tf.placeholder(tf.float32, shape=[None, None, 4])  # gt_boxes
    gt_labels = tf.placeholder(tf.int64, shape=[None, None])  # gt_labels
    orig_gt_counts = tf.placeholder(tf.int32, shape=[None, ])  # orig_gt_counts
    anchor_gt_labels = tf.placeholder(tf.int32, shape=[None, None, None, None])  # anchor_gt_labels
    anchor_gt_boxes = tf.placeholder(tf.float32, shape=[None, None, None, None, 4])  # anchor_gt_boxes

    sess_config = tf.ConfigProto()
    sess_config.allow_soft_placement = True
    sess_config.log_device_placement = False
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    print('begin data loading...')
    start = time.time()
    for step in range(10000000000000000):
        blobs = next(data_iter)
        images_, gt_boxes_, gt_labels_, orig_gt_counts_, anchor_labels_, anchor_boxes_ = \
            sess.run([images, gt_boxes, gt_labels, orig_gt_counts, anchor_gt_labels, anchor_gt_boxes],
                     feed_dict={
                         images: blobs['images'],
                         gt_boxes: blobs['gt_boxes'],
                         gt_labels: blobs['gt_labels'],
                         orig_gt_counts: blobs['orig_gt_counts'],
                         anchor_gt_labels: blobs['anchor_labels'],
                         anchor_gt_boxes: blobs['anchor_boxes']
                     })

        if step %20 == 0:

            print(step, (time.time() - start) / 20)
            start = time.time()


if __name__ == '__main__1':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import time
    import tensorflow as tf
    from dataset import register_coco, register_pascal_voc

    num_gpus = 2

    register_coco(os.path.expanduser("/media/ubuntu/Working/coco"))
    # register_pascal_voc(os.path.expanduser("/media/ubuntu/Working/voc2012/VOC2012/"))

    print('get dataflow ...')

    data_iter = get_train_dataflow(batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * num_gpus)
    ds = tf.data.Dataset.from_generator(lambda: map(lambda x: tuple([x[k] for k in
                                                        ['images', 'gt_boxes', 'gt_labels', 'orig_gt_counts',
                                                         'anchor_labels_level2', 'anchor_boxes_level2',
                                                         'anchor_labels_level3', 'anchor_boxes_level3',
                                                         'anchor_labels_level4', 'anchor_boxes_level4',
                                                         'anchor_labels_level5', 'anchor_boxes_level5',
                                                         'anchor_labels_level6', 'anchor_boxes_level6']]),
                                       data_iter),
                           (tf.float32, tf.float32, tf.int64, tf.int32,
                            tf.int32, tf.float32,
                            tf.int32, tf.float32,
                            tf.int32, tf.float32,
                            tf.int32, tf.float32,
                            tf.int32, tf.float32),
                           (tf.TensorShape([None, None, None, 3]),
                            tf.TensorShape([None, None, 4]),
                            tf.TensorShape([None, None]),
                            tf.TensorShape([None, ]),
                            tf.TensorShape([None, None, None, None]), tf.TensorShape([None, None, None, None, 4]), #lv2
                            tf.TensorShape([None, None, None, None]), tf.TensorShape([None, None, None, None, 4]), #lv3
                            tf.TensorShape([None, None, None, None]), tf.TensorShape([None, None, None, None, 4]), #lv4
                            tf.TensorShape([None, None, None, None]), tf.TensorShape([None, None, None, None, 4]), #lv5
                            tf.TensorShape([None, None, None, None]), tf.TensorShape([None, None, None, None, 4])  #lv6
                            ))
    ds = ds.prefetch(buffer_size=128)
    ds = ds.make_one_shot_iterator()
    images, gt_boxes, gt_labels, orig_gt_counts, \
    anchor_labels_level2, anchor_boxes_level2, \
    anchor_labels_level3, anchor_boxes_level3, \
    anchor_labels_level4, anchor_boxes_level4, \
    anchor_labels_level5, anchor_boxes_level5, \
    anchor_labels_level6, anchor_boxes_level6 \
    = ds.get_next()
    fpn_anchor_gt_labels = \
        [anchor_labels_level2, anchor_labels_level3, anchor_labels_level4, anchor_labels_level5, anchor_labels_level6]
    fpn_anchor_gt_boxes = \
        [anchor_boxes_level2, anchor_boxes_level3, anchor_boxes_level4, anchor_boxes_level5, anchor_boxes_level6]

    sess_config = tf.ConfigProto()
    sess_config.allow_soft_placement = True
    sess_config.log_device_placement = False
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    print('begin data loading...')
    start = time.time()
    for step in range(10000000000000000):
        # blobs = next(data_iter)
        images_, gt_boxes_, gt_labels_, orig_gt_counts_, \
            anchor_labels_level2_, anchor_boxes_level2_, \
            anchor_labels_level3_, anchor_boxes_level3_, \
            anchor_labels_level4_, anchor_boxes_level4_, \
            anchor_labels_level5_, anchor_boxes_level5_, \
            anchor_labels_level6_, anchor_boxes_level6_= \
            sess.run([images, gt_boxes, gt_labels, orig_gt_counts, anchor_labels_level2, anchor_boxes_level2,
                      anchor_labels_level3, anchor_boxes_level3,
                      anchor_labels_level4, anchor_boxes_level4,
                      anchor_labels_level5, anchor_boxes_level5,
                      anchor_labels_level6, anchor_boxes_level6])

        if step %20 == 0:

            print(step, (time.time() - start) / 20)
            print(images_.shape)
            print(anchor_labels_level2_.shape, anchor_boxes_level2_.shape)
            print(anchor_labels_level3_.shape, anchor_boxes_level3_.shape)
            print(anchor_labels_level4_.shape, anchor_boxes_level4_.shape)
            print(anchor_labels_level5_.shape, anchor_boxes_level5_.shape)
            print(anchor_labels_level6_.shape, anchor_boxes_level6_.shape)
            start = time.time()
