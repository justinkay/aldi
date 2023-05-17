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

    # EMA of student weights
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
    _C.MODEL.SADA = CN()
    _C.MODEL.SADA.ENABLED = False
    _C.MODEL.SADA.DA_IMG_GRL_WEIGHT = 0.01
    _C.MODEL.SADA.DA_INS_GRL_WEIGHT = 0.1
    _C.MODEL.SADA.COS_WEIGHT = 0.1

    # Probabilistic Teacher (Gaussian RCNN) settings
    _C.GRCNN = CN()
    _C.GRCNN.LEARN_ANCHORS_LABELED = False
    _C.GRCNN.LEARN_ANCHORS_UNLABELED = False
    _C.GRCNN.TAU = [0.5, 0.5]
    _C.GRCNN.EFL = False
    _C.GRCNN.EFL_LAMBDA = [0.5, 0.5]
    _C.GRCNN.MODEL_TYPE = "GAUSSIAN"
    # TODO: Where did they get these?
    _C.MODEL.ANCHOR_GENERATOR.ANCHOR = [[[181.0193, 90.5097],
                                        [128.0000, 128.0000],
                                        [90.5097, 181.0193],
                                        [362.0387, 181.0193],
                                        [256.0000, 256.0000],
                                        [181.0193, 362.0387],
                                        [724.0773, 362.0387],
                                        [512.0000, 512.0000],
                                        [362.0387, 724.0773]], ]