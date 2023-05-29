# Everything in this file makes domain adaptation disabled by default.
# Everything must be explicitly enabled in the config files.

from detectron2.config import CfgNode as CN


def add_da_config(cfg):
    _C = cfg

    _C.DOMAIN_ADAPT = CN()

    # Datasets and sampling
    _C.DATASETS.UNLABELED = tuple()
    _C.DATASETS.BATCH_CONTENTS = ("labeled_weak", ) # one or more of: { "labeled_weak", "labeled_strong", "unlabeled_weak", "unlabeled_strong" }
    _C.DATASETS.BATCH_RATIOS = (1,) # must match length of BATCH_CONTENTS

    # Strong augmentations
    _C.AUG = CN()
    _C.AUG.INCLUDE_RANDOM_ERASING = True
    _C.AUG.LABELED_MIC_AUG = False
    _C.AUG.UNLABELED_MIC_AUG = False
    _C.AUG.MIC_RATIO = 0.5
    _C.AUG.MIC_BLOCK_SIZE = 32

    # EMA of student weights
    _C.EMA = CN()
    _C.EMA.ENABLED = False
    _C.EMA.ALPHA = 0.9996 # From Adaptive Teacher settings
    _C.EMA.PARALLEL = False # Whether to use model parallelism (put EMA model on a different GPU than the student model). Note this only works with 2 GPUs, and does not work with DDP (must launch with --num-gpus=1)

    # Teacher model provides pseudo labels
    _C.DOMAIN_ADAPT.TEACHER = CN()
    _C.DOMAIN_ADAPT.TEACHER.PSEUDO_LABEL_METHOD = "thresholding" # one of: { "thresholding", "probabilistic" }
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

    # Adaptive Teacher style adverarial feature alignment
    _C.MODEL.DA = CN()
    _C.MODEL.DA.ENABLED = False
    _C.MODEL.DA.DIS_TYPE = "p2"
    _C.MODEL.DA.DIS_LOSS_WEIGHT = 0.05

    # Probabilistic Teacher (Gaussian RCNN) settings
    _C.GRCNN = CN()
    _C.GRCNN.LEARN_ANCHORS_LABELED = False
    _C.GRCNN.LEARN_ANCHORS_UNLABELED = False
    _C.GRCNN.TAU = [0.5, 0.5]
    _C.GRCNN.EFL = False
    _C.GRCNN.EFL_LAMBDA = [0.5, 0.5]
    _C.GRCNN.MODEL_TYPE = "GAUSSIAN"

    # We use gradient accumulation to run the weak/strong/unlabeled data separately
    # Should we call backward intermittently during accumulation or at the end?
    # The former is slower but less memory usage
    _C.SOLVER.BACKWARD_AT_END = True

    # Enable use of different optimizers (necessary to match VitDet settings)
    _C.SOLVER.OPTIMIZER = "SGD"