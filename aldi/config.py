# Everything in this file makes domain adaptation disabled by default.
# Everything must be explicitly enabled in the config files.

from detectron2.config import CfgNode as CN


def add_aldi_config(cfg):
    _C = cfg

    # Datasets and sampling
    _C.DATASETS.UNLABELED = tuple()
    _C.DATASETS.BATCH_CONTENTS = ("labeled_weak", ) # one or more of: { "labeled_weak", "labeled_strong", "unlabeled_weak", "unlabeled_strong" }
    _C.DATASETS.BATCH_RATIOS = (1,) # must match length of BATCH_CONTENTS

    # Strong augmentations
    _C.AUG = CN()
    _C.AUG.WEAK_INCLUDES_MULTISCALE = True
    _C.AUG.LABELED_INCLUDE_RANDOM_ERASING = True
    _C.AUG.UNLABELED_INCLUDE_RANDOM_ERASING = True
    _C.AUG.LABELED_MIC_AUG = False
    _C.AUG.UNLABELED_MIC_AUG = False
    _C.AUG.MIC_RATIO = 0.5
    _C.AUG.MIC_BLOCK_SIZE = 32

    # EMA of student weights
    _C.EMA = CN()
    _C.EMA.ENABLED = False
    _C.EMA.ALPHA = 0.9996
     # when loading a model at the start of training (i.e. not resuming mid-training run),
     # if MODEL.WEIGHTS contains both ["model", "ema"], initialize with the EMA weights.
     # also determines if EMA is used for eval when running tools/train_net.py --eval-only.
    _C.EMA.LOAD_FROM_EMA_ON_START = True
    _C.EMA.START_ITER = 0

    # Begin domain adaptation settings
    _C.DOMAIN_ADAPT = CN()

    # Source-target alignment
    _C.DOMAIN_ADAPT.ALIGN = CN()
    _C.DOMAIN_ADAPT.ALIGN.MIXIN_NAME = "AlignMixin"
    _C.DOMAIN_ADAPT.ALIGN.IMG_DA_ENABLED = False
    _C.DOMAIN_ADAPT.ALIGN.IMG_DA_LAYER = "p2"
    _C.DOMAIN_ADAPT.ALIGN.IMG_DA_WEIGHT = 0.01
    _C.DOMAIN_ADAPT.ALIGN.IMG_DA_INPUT_DIM = 256 # = output channels of backbone
    _C.DOMAIN_ADAPT.ALIGN.IMG_DA_HIDDEN_DIMS = [256,]
    _C.DOMAIN_ADAPT.ALIGN.INS_DA_ENABLED = False
    _C.DOMAIN_ADAPT.ALIGN.INS_DA_WEIGHT = 0.01
    _C.DOMAIN_ADAPT.ALIGN.INS_DA_INPUT_DIM = 1024 # = output channels of box head
    _C.DOMAIN_ADAPT.ALIGN.INS_DA_HIDDEN_DIMS = [1024,]

    # Self-distillation
    _C.DOMAIN_ADAPT.DISTILL = CN()
    _C.DOMAIN_ADAPT.DISTILL.DISTILLER_NAME = "ALDIDistiller"
    _C.DOMAIN_ADAPT.DISTILL.MIXIN_NAME = "DistillMixin"
    # 'Pseudo label' approaches
    _C.DOMAIN_ADAPT.DISTILL.HARD_ROIH_CLS_ENABLED = False
    _C.DOMAIN_ADAPT.DISTILL.HARD_ROIH_REG_ENABLED = False
    _C.DOMAIN_ADAPT.DISTILL.HARD_OBJ_ENABLED = False
    _C.DOMAIN_ADAPT.DISTILL.HARD_RPN_REG_ENABLED = False
    # 'Distillation' approaches
    _C.DOMAIN_ADAPT.DISTILL.ROIH_CLS_ENABLED = False
    _C.DOMAIN_ADAPT.DISTILL.ROIH_REG_ENABLED = False
    _C.DOMAIN_ADAPT.DISTILL.OBJ_ENABLED = False
    _C.DOMAIN_ADAPT.DISTILL.RPN_REG_ENABLED = False
    _C.DOMAIN_ADAPT.DISTILL.CLS_TMP = 1.0
    _C.DOMAIN_ADAPT.DISTILL.OBJ_TMP = 1.0
    _C.DOMAIN_ADAPT.CLS_LOSS_TYPE = "CE" # one of: { "CE", "KL" }

    # Teacher model provides pseudo labels
    # TODO: Could be merged into DISTILL settings somehow
    _C.DOMAIN_ADAPT.TEACHER = CN()
    _C.DOMAIN_ADAPT.TEACHER.ENABLED = False
    _C.DOMAIN_ADAPT.TEACHER.THRESHOLD = 0.8

    # Vision Transformer settings
    _C.VIT = CN()
    _C.VIT.USE_ACT_CHECKPOINT = True

    # We interpret SOLVER.IMS_PER_BATCH as the total batch size on all GPUs, for 
    # experimental consistency. Gradient accumulation is used according to 
    # num_gradient_accum_steps = IMS_PER_BATCH / (NUM_GPUS * IMS_PER_GPU)
    _C.SOLVER.IMS_PER_GPU = 2

    # We use gradient accumulation to run the weak/strong/unlabeled data separately
    # Should we call backward intermittently during accumulation or at the end?
    # The former is slower but less memory usage
    _C.SOLVER.BACKWARD_AT_END = True

    # Enable use of different optimizers (necessary to match VitDet settings)
    _C.SOLVER.OPTIMIZER = "SGD"

    # Extra configs for convnext
    # Default is ConvNext-T (Resnet-50 equiv.)
    _C.MODEL.CONVNEXT = CN()
    _C.MODEL.CONVNEXT.DEPTHS= [3, 3, 9, 3]
    _C.MODEL.CONVNEXT.DIMS= [96, 192, 384, 768]
    _C.MODEL.CONVNEXT.DROP_PATH_RATE= 0.2
    _C.MODEL.CONVNEXT.LAYER_SCALE_INIT_VALUE= 1e-6
    _C.MODEL.CONVNEXT.OUT_FEATURES= [0, 1, 2, 3]
    _C.SOLVER.WEIGHT_DECAY_RATE= 0.95