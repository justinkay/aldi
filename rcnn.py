import torch
from typing import Dict, List

from detectron2.config import configurable
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

@META_ARCH_REGISTRY.register()
class DARCNN(GeneralizedRCNN):

    @configurable
    def __init__(
        self,
        *,
        do_reg_loss: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.do_reg_loss = do_reg_loss

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret.update({"do_reg_loss": cfg.DOMAIN_ADAPT.LOSSES.RPN_LOSS_ENABLED})
        return ret

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        output = super().forward(batched_inputs)
        # Some methods (Adaptive/Unbiased Teacher) disable the regression losses
        if self.training and not self.do_reg_loss:
            output["loss_rpn_loc"] *= 0
            output["loss_box_reg"] *= 0
        return output