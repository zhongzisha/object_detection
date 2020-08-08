# -*- coding: utf-8 -*-
# @Time    : 2019/11/8 15:09
# @Author  : Zisha Zhong
# @Email   : zhongzisha@outlook.com
# @File    : myaug_lib.py
# @Software: PyCharm

import numpy as np
import cv2
import random


def short_side_resize_image(img, boxes=None, short_side_length=800, max_size=1333):
    h, w = img.shape[:2]
    scale = short_side_length * 1.0 / min(h, w)
    if h < w:
        newh, neww = short_side_length, scale * w
    else:
        newh, neww = scale * h, short_side_length
    if max(newh, neww) > max_size:
        scale = max_size * 1.0 / max(newh, neww)
        newh = newh * scale
        neww = neww * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)

    if boxes is not None:
        boxes[:, [0, 2]] *= (neww * 1.0 / w)
        boxes[:, [1, 3]] *= (newh * 1.0 / h)

        return cv2.resize(img, (neww, newh)), boxes.astype(np.int32)

    return cv2.resize(img, (neww, newh))


def random_horizontal_flip(im, boxes=None):

    h, w = im.shape[:2]
    if random.random() < 0.5:
        im = im[:, ::-1, :]
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

    return im, boxes


def random_vertical_flip(im, boxes=None):

    h, w = im.shape[:2]
    if random.random() < 0.5:
        im = im[::-1, :, :]
        boxes[:, [1, 3]] = h - boxes[:, [3, 1]]

    return im, boxes


