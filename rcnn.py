import torch
from typing import Dict, List

from detectron2.config import configurable
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.layers import cat, cross_entropy

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
        do_reg_loss_unlabeled: bool = True,
        do_quality_loss_weight_unlabeled: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.do_reg_loss_unlabeled = do_reg_loss_unlabeled
        self.do_quality_loss_weight_unlabeled = do_quality_loss_weight_unlabeled

        # register hooks so we can grab output of sub-modules
        self.rpn_io, self.roih_io, self.boxpred_io = SaveIO(), SaveIO(), SaveIO()
        self.proposal_generator.register_forward_hook(self.rpn_io)
        self.roi_heads.register_forward_hook(self.roih_io)
        self.roi_heads.box_predictor.register_forward_hook(self.boxpred_io)

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret.update({"do_reg_loss_unlabeled": cfg.DOMAIN_ADAPT.LOSSES.LOC_LOSS_ENABLED})
        ret.update({"do_quality_loss_weight_unlabeled": cfg.DOMAIN_ADAPT.LOSSES.QUALITY_LOSS_WEIGHT_ENABLED})
        return ret

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], labeled=True):
        output = super().forward(batched_inputs)

        if self.training and not labeled:
            # Some methods (Adaptive/Unbiased Teacher, MIC) disable the regression losses
            if not self.do_reg_loss_unlabeled:
                output["loss_rpn_loc"] *= 0
                output["loss_box_reg"] *= 0

            # Others weight classification losses by "quality" (implemented in MIC)
            if self.do_quality_loss_weight_unlabeled:
                proposals, _ = self.roih_io.output
                predictions = self.boxpred_io.output
                scores, _ = predictions
                gt_classes = (cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0))
                quality = torch.max(torch.softmax(scores, dim=1), dim=1)[0]
                loss_cls = torch.mean(quality * cross_entropy(scores, gt_classes, reduction="none"))
                output["loss_cls"] = loss_cls * self.roi_heads.box_predictor.loss_weight.get("loss_cls", 1.0)

        return output