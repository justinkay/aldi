# register all detectron2 componenets
from .anchor_generator import DifferentiableAnchorGenerator
from .roi_heads import GaussianROIHead
from .rpn import GaussianRPN, GaussianRPNHead