import torch
from typing import Dict, List

from detectron2.config import configurable
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY


class SaveIO:
    """Simple PyTorch hook to save the output of a nn.module."""
    def __init__(self):
        self.input = None
        self.output = None
        
    def __call__(self, module, module_in, module_out):
        self.input = module_in
        self.output = module_out

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

        # register hooks so we can grab output of sub-modules
        self.rpn_io, self.roih_io = SaveIO(), SaveIO()
        self.proposal_generator.register_forward_hook(self.rpn_io)
        self.roi_heads.register_forward_hook(self.roih_io)

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