# -*- coding: utf-8 -*-
# @Time    : 2019/11/7 16:39
# @Author  : Zisha Zhong
# @Email   : zhongzisha@outlook.com
# @File    : viz.py
# @Software: PyCharm


import numpy as np
import tensorflow as tf
import tfplot as tfp
import cv2

from config import cfg

MYPALETTE_RGB = np.random.randint(low=0, high=255, size=(256, 3)).astype(np.float32)


def draw_boxes(img, boxes, labels, prefix='none', color=(0, 255, 0)):

    # img = cv2.cvtColor(np.copy(img), cv2.COLOR_BGR2RGB)
    img = img.copy()
    if boxes is not None and boxes.shape[0] > 0:
        print('{} has {} boxes'.format(prefix, boxes.shape[0]))
        boxes = np.copy(boxes).astype(np.int32)
        for idx in range(boxes.shape[0]):
            xmin, ymin, xmax, ymax = boxes[idx, :]
            label = int(labels[idx])
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=color, thickness=2)
            cv2.putText(img, text='{}'.format(label), org=(xmin, ymin),
                        fontFace=1, fontScale=1, color=(0, 0, 255))
    else:
        print('{} no boxes'.format(prefix), img.shape)
    return img


def draw_boxes_with_scores(img, boxes, labels, scores, prefix='none', color=(0, 255, 0)):

    # img = cv2.cvtColor(np.copy(img), cv2.COLOR_BGR2RGB)
    img = img.copy()
    if boxes is not None and boxes.shape[0] > 0:
        print('{} has {} boxes'.format(prefix, boxes.shape[0]))
        boxes = np.copy(boxes).astype(np.int32)
        scores = np.copy(scores).astype(np.float32)
        num = min(len(scores), 10)  # only show 10 samples
        sorted_ids = np.argsort(scores)[::-1][:num]
        boxes = boxes[sorted_ids, :]
        scores = scores[sorted_ids]
        labels = labels[sorted_ids]
        print('-'*50)
        for idx in range(boxes.shape[0]):
            xmin, ymin, xmax, ymax = boxes[idx, :]
            label = int(labels[idx])
            score = scores[idx]
            print('{:d},{:d},{:d},{:d},{:.3f},{:d}'.format(xmin,ymin,xmax,ymax,score,label))
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=color, thickness=2)
            cv2.putText(img, text='{}:{:.3f}'.format(label, score), org=(xmin, ymin),
                        fontFace=1, fontScale=1, color=(0, 0, 255))
    else:
        print('{} no boxes'.format(prefix), img.shape)
    return img


def draw_on_img(feat_h, feat_w, locs, vals):
    def draw_routine(feat_h_, feat_w_, locs_, vals_):
        locs_ = locs_.astype(np.int32)
        img = np.zeros((feat_w_, feat_h_), dtype=np.uint8)
        if len(locs_) > 0:
            vals_ = vals_.astype(np.float32)
            vals_ = (vals_ - vals_.min()) / max((vals_.max() - vals_.min(), 1)) * 255
            for idx in range(locs_.shape[0]):
                x = locs_[idx, 0]
                y = locs_[idx, 1]
                x1, x2 = max(x - 10, 1), min(x + 10, feat_w_ - 1)
                y1, y2 = max(y - 10, 1), min(y + 10, feat_h_ - 1)
                img[x1:x2, y1:y2] = vals_[idx]
        img[img > 255] = 255
        return img

    img_t = tf.py_func(draw_routine, inp=[feat_h, feat_w, locs, vals], Tout=[tf.uint8])
    return img_t


def draw_on_img_with_color(feat_h, feat_w, locs, vals):
    def draw_routine(feat_h_, feat_w_, locs_, vals_):
        locs_ = locs_.astype(np.int32)
        vals_ = vals_.astype(np.uint8)
        img = np.zeros((feat_w_, feat_h_), dtype=np.uint8)
        if len(locs_) > 0:
            for idx in range(locs_.shape[0]):
                x = locs_[idx, 0]
                y = locs_[idx, 1]
                x1, x2 = max(x - 10, 1), min(x + 10, feat_w_ - 1)
                y1, y2 = max(y - 10, 1), min(y + 10, feat_h_ - 1)
                img[x1:x2, y1:y2] = vals_[idx]
            img_color = MYPALETTE_RGB[img].astype(np.uint8)
            # print('img_color shape', img_color.shape)
        else:
            img_color = np.zeros((feat_w_, feat_h_, 3), dtype=np.uint8)
        return img_color

    img_t = tf.py_func(draw_routine, inp=[feat_h, feat_w, locs, vals], Tout=[tf.uint8])
    return img_t


def figure_attention(activation):
    fig, ax = tfp.subplots()
    im = ax.imshow(activation, cmap='jet')
    fig.colorbar(im)
    return fig


def draw_heatmap(feature_map, name, data_format='channels_first'):
    if data_format == 'channels_first':
        feature_map = tf.transpose(feature_map, [0, 2, 3, 1])
    for b_id in range(cfg.TRAIN.BATCH_SIZE_PER_GPU):
        heatmap = tf.reduce_sum(feature_map[b_id], axis=-1)
        tfp.summary.plot('image{}_{}'.format(b_id, name), figure_attention, [heatmap])
