# Everything in this file makes domain adaptation disabled by default.
# Everything must be explicitly enabled in the config files.

from detectron2.config import CfgNode as CN


def add_da_config(cfg):
    _C = cfg

    _C.DOMAIN_ADAPT = CN()

    # Datasets and sampling
    _C.DATASETS.UNLABELED = tuple()
    _C.DATASETS.LABELED_UNLABELED_RATIO = (1,0)
    _C.DATASETS.LABELED_STRONG_AUG = False
    _C.DATASETS.UNLABELED_STRONG_AUG = False
    _C.DATASETS.INCLUDE_WEAK_IN_BATCH = False
    _C.DATASETS.INCLUDE_RANDOM_ERASING = True

    _C.DATASETS.LABELED_MIC_AUG = False
    _C.DATASETS.UNLABELED_MIC_AUG = False
    _C.DATASETS.MIC_RATIO = 0.5
    _C.DATASETS.MIC_BLOCK_SIZE = 32

    # EMA of stuent weights
    _C.EMA = CN()
    _C.EMA.ENABLED = False
    _C.EMA.ALPHA = 0.9996 # From Adaptive Teacher settings

    # Teacher model provides pseudo labels
    _C.DOMAIN_ADAPT.TEACHER = CN()
    _C.DOMAIN_ADAPT.TEACHER.PSEUDO_LABEL_METHOD = "thresholding" # could support a tuple multiple methods in the future
    _C.DOMAIN_ADAPT.TEACHER.THRESHOLD = 0.8

    # Custom loss functions/modifications
    _C.DOMAIN_ADAPT.LOSSES = CN()
    _C.DOMAIN_ADAPT.LOSSES.LOC_LOSS_ENABLED = True
    _C.DOMAIN_ADAPT.LOSSES.QUALITY_LOSS_WEIGHT_ENABLED = False

    # SADA settings
    _C.MODEL.DA_HEADS = CN()
    _C.MODEL.DA_HEADS.ENABLED = False
    _C.MODEL.DA_HEADS.DA_IMG_GRL_WEIGHT = 0.01
    _C.MODEL.DA_HEADS.DA_INS_GRL_WEIGHT = 0.1
    _C.MODEL.DA_HEADS.COS_WEIGHT = 0.1