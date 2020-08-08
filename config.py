# -*- coding: utf-8 -*-
# @Time    : 2019/11/7 14:10
# @Author  : Zisha Zhong
# @Email   : zhongzisha@outlook.com
# @File    : config.py
# @Software: PyCharm

from easydict import EasyDict as edict

cfg = edict()

cfg.VISUALIZATION = True

cfg.MODE_FRCNN = True
cfg.MODE_FPN = False
cfg.MODE_MASK = False
cfg.MODE_FCOS = False
cfg.MODE_RETINANET = False

# for BACKBONE
cfg.BACKBONE = edict()
cfg.BACKBONE.WEIGHTS = ''
cfg.BACKBONE.DATA_FORMAT = 'channels_first'
cfg.BACKBONE.STRIDE_IN_1x1 = True  # for MSRA model, this is True
cfg.BACKBONE.BATCH_NORM_DECAY = 0.997
cfg.BACKBONE.BATCH_NORM_EPSILON = 1e-4
cfg.BACKBONE.CHECKPOINT_PATH = 'F:/FCOS/resnet_resnet-nhwc-2018-02-07/model.ckpt-112603'

# for FPN
cfg.FPN = edict()
cfg.FPN.STRIDES = [4, 8, 16, 32, 64]
# build FPN based on c2, c3, c4, c5, for FPN, should be 64
# for RetinaNet, should be 128
cfg.FPN.RESOLUTION_REQUIREMENT = 128  # p3, p4, p5, p6, p7

#-------------------------------
# for FasterRCNN
cfg.FRCNN = edict()
cfg.FRCNN.VISUALIZATION = True
cfg.FRCNN.ANCHOR = edict()
cfg.FRCNN.ANCHOR.STRIDE = 16
cfg.FRCNN.ANCHOR.SIZES = (32, 64, 128, 256, 512)
cfg.FRCNN.ANCHOR.RATIOS = (0.5, 1., 2.0)
#-------------------------------
cfg.FRCNN.RPN = edict()
cfg.FRCNN.RPN.CHANNELS = 256  # 1024
cfg.FRCNN.RPN.FG_RATIO = 0.5
cfg.FRCNN.RPN.BATCH_PER_IM = 256
cfg.FRCNN.RPN.POSITIVE_ANCHOR_THRESH = 0.7
cfg.FRCNN.RPN.NEGATIVE_ANCHOR_THRESH = 0.3
# Anchors which overlap with a crowd box (IOA larger than threshold) will be ignored.
# Setting this to a value larger than 1.0 will disable the feature.
# It is disabled by default because Detectron does not do this.
cfg.FRCNN.RPN.CROWD_OVERLAP_THRESH = 9.99
cfg.FRCNN.RPN.TRAIN_PRE_NMS_TOPK = 12000
cfg.FRCNN.RPN.TRAIN_POST_NMS_TOPK = 2000
cfg.FRCNN.RPN.TEST_PRE_NMS_TOPK = 6000
cfg.FRCNN.RPN.TEST_POST_NMS_TOPK = 1000
cfg.FRCNN.RPN.TRAIN_PER_LEVEL_NMS_TOPK = 2000
cfg.FRCNN.RPN.TEST_PER_LEVEL_NMS_TOPK = 1000
cfg.FRCNN.RPN.MIN_SIZE = 0
cfg.FRCNN.RPN.PROPOSAL_NMS_THRESH = 0.7
#-------------------------------
cfg.FRCNN.FPN = edict()
cfg.FRCNN.FPN.ANCHOR_STRIDES = (4, 8, 16, 32, 64)  # p2, p3, p4, p5, p6
cfg.FRCNN.FPN.NUM_CHANNEL = 256
#-------------------------------
cfg.FRCNN.RCNN = edict()
cfg.FRCNN.RCNN.BATCH_PER_IM = 512
cfg.FRCNN.RCNN.FG_THRESH = 0.5
cfg.FRCNN.RCNN.FG_RATIO = 0.25
cfg.FRCNN.RCNN.BBOX_REG_WEIGHTS = [10., 10., 5., 5.]
#-------------------------------
cfg.FRCNN.TEST = edict()
cfg.FRCNN.TEST.RESULT_SCORE_THRESH = 0.05
cfg.FRCNN.TEST.NMS_THRESH = 0.5
cfg.FRCNN.TEST.RESULTS_PER_IM = 100

# for FCOS
cfg.FCOS = edict()
cfg.FCOS.NORM_REG_TARGETS = False
cfg.FCOS.CENTERNESS_ON_REG = False
cfg.FCOS.PRIOR_PROB = 0.01
cfg.FCOS.IOU_LOSS_TYPE = 'linear_iou'  # iou, linear_iou, giou
cfg.FCOS.VISUALIZATION = True
cfg.FCOS.FPN_STRIDES = (8, 16, 32, 64, 128)
cfg.FCOS.PRE_NMS_THRESH = 0.05
cfg.FCOS.PRE_NMS_TOP_N = 1000
cfg.FCOS.NMS_THRESH = 0.6
cfg.FCOS.FPN_POST_NMS_TOP_N = 100

# for RetinaNet
cfg.RETINANET = edict()
cfg.RETINANET.ANCHOR_SIZES = (32, 64, 128, 256, 512)  # p3,p4,p5,p6,p7
cfg.RETINANET.ANCHOR_STRIDES = (8, 16, 32, 64, 128)   # p3,p4,p5,p6,p7
cfg.RETINANET.ANCHOR_SCALES = (2**0, 2**(1.0/3.0), 2**(2.0/3.0))
cfg.RETINANET.ANCHOR_RATIOS = (0.5, 1, 2.0)
cfg.RETINANET.TRAIN_PER_LEVEL_NMS_TOPK = 2000
cfg.RETINANET.TEST_PER_LEVEL_NMS_TOPK = 1000
cfg.RETINANET.POSITIVE_ANCHOR_THRESH = 0.5   # [0.5, 1.0]  positive samples
cfg.RETINANET.NEGATIVE_ANCHOR_THRESH = 0.4   # [0, 0.4)    negative samples

cfg.DATA = edict()
cfg.DATA.DATASET_NAME = 'COCO'
cfg.DATA.BASEDIR = 'F:/coco/'
cfg.DATA.NOAUG_FILENAME = ''
cfg.DATA.NUM_CATEGORY = 80  # no background class
cfg.DATA.TRAIN = ('coco_train2017',)
cfg.DATA.TEST = ('coco_val2017',)
cfg.DATA.NUM_WORKERS = 2
cfg.DATA.ASPECT_RATIO_GROUPING = True

cfg.TRAIN = edict()
cfg.TRAIN.BASE_LR = 0.000625   # 0.01 for 16, 0.005 for 8, 0.0025 for 4, 0.00125 for 2, 0.000625 for 1
cfg.TRAIN.LR_BOUNDARIES = [480000, 640000]  #
cfg.TRAIN.MAX_STEPS = 720000  # BS=16
cfg.TRAIN.BATCH_SIZE_PER_GPU = 2
cfg.TRAIN.GPU_LIST = '0'
cfg.TRAIN.NUM_GPUS = len(cfg.TRAIN.GPU_LIST.split(','))
cfg.TRAIN.SAVE_SUMMARY_STEPS = 200
cfg.TRAIN.STEPS_PER_EPOCH = 500
cfg.TRAIN.WARMUP_STEP = 500
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.LOG_DIR = 'E:/objectdetection/logs_{}GPUs_BS={}_LR={}/'.format(
    cfg.TRAIN.NUM_GPUS, cfg.TRAIN.BATCH_SIZE_PER_GPU, cfg.TRAIN.BASE_LR)
cfg.TRAIN.MOVING_AVERAGE_DECAY = 0.997

cfg.PREPROC = edict()
# cfg.PREPROC.PIXEL_MEAN = [103.53, 116.28, 123.675]  # BGR format
# cfg.PREPROC.PIXEL_STD = [1., 1., 1.]  # [57.375, 57.12, 58.395]
cfg.PREPROC.PIXEL_MEAN = [0.485, 0.456, 0.406] # [103.53, 116.28, 123.675]  # BGR format
cfg.PREPROC.PIXEL_STD = [0.229, 0.224, 0.225]  # [1., 1., 1.] #[57.375, 57.12, 58.395]
cfg.PREPROC.PIXEL_SCALE = 255
cfg.PREPROC.PREDEFINED_PADDING = False
cfg.PREPROC.PADDING_SHAPES = [(800, 1000), (800, 1200)]
cfg.PREPROC.TRAIN_SHORT_EDGE_SIZE = [800, 800]  # [min, max] to sample from
cfg.PREPROC.TEST_SHORT_EDGE_SIZE = 800
cfg.PREPROC.MAX_SIZE = 1333

cfg.PREPROC.PADDING_HEIGHT = 240
cfg.PREPROC.PADDING_WIDTH = 240
cfg.PREPROC.AUG_TYPE = 'online' # or 'offline'
cfg.PREPROC.AUG_COLOR2GRAY = False

