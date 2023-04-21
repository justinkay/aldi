# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_da_config(cfg):
    _C = cfg

    _C.DOMAIN_ADAPT = CN()

    # EMA of stuent weights
    _C.DOMAIN_ADAPT.EMA = CN()
    _C.DOMAIN_ADAPT.EMA.ENABLED = False
    _C.DOMAIN_ADAPT.EMA.ALPHA = 0.9996 # From Adaptive Teacher settings

    # Teacher model provides pseudo labels
    _C.DOMAIN_ADAPT.TEACHER = CN()
    _C.DOMAIN_ADAPT.TEACHER.ENABLED = False
    _C.DOMAIN_ADAPT.TEACHER.PSEUDO_LABEL_METHOD = "thresholding" # could support a tuple multiple methods in the future
    _C.DOMAIN_ADAPT.TEACHER.THRESHOLD = 0.8

    # From Adaptive Teacher - to be replaced
    _C.SOLVER.IMG_PER_BATCH_LABEL = 4
    _C.SOLVER.IMG_PER_BATCH_UNLABEL = 4
    _C.DATASETS.TRAIN_LABEL = tuple()
    _C.DATASETS.TRAIN_UNLABEL = tuple()

    