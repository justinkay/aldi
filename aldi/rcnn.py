import torch
from typing import Dict, List

from detectron2.config import configurable
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.layers import cat, cross_entropy

from align import AlignMixin
from distill import DistillMixin


@META_ARCH_REGISTRY.register()
class ALDI(AlignMixin, DistillMixin, GeneralizedRCNN): 
    @configurable
    def __init__(self, **kwargs):
        super(ALDI, self).__init__(**kwargs)

    @classmethod
    def from_config(cls, cfg):
        return super(ALDI, cls).from_config(cfg)

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], 
                labeled: bool = True, do_align: bool = False):
        return super(ALDI, self).forward(batched_inputs, do_align=do_align, labeled=labeled)