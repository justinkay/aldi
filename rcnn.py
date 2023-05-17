import torch
from typing import Dict, List, Bool

from detectron2.config import configurable
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.layers import cat, cross_entropy

from sada import SADA

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
        sada_heads: SADA = None,
        **kwargs
    ):
        super(DARCNN, self).__init__(**kwargs)
        self.do_reg_loss_unlabeled = do_reg_loss_unlabeled
        self.do_quality_loss_weight_unlabeled = do_quality_loss_weight_unlabeled
        self.sada_heads = sada_heads

        # register hooks so we can grab output of sub-modules
        self.backbone_io, self.rpn_io, self.roih_io, self.boxhead_io, self.boxpred_io = SaveIO(), SaveIO(), SaveIO(), SaveIO(), SaveIO()
        self.backbone.register_forward_hook(self.backbone_io)
        self.proposal_generator.register_forward_hook(self.rpn_io)
        self.roi_heads.register_forward_hook(self.roih_io)
        self.roi_heads.box_head.register_forward_hook(self.boxhead_io)
        self.roi_heads.box_predictor.register_forward_hook(self.boxpred_io)

    @classmethod
    def from_config(cls, cfg):
        ret = super(DARCNN, cls).from_config(cfg)

        # loss modifications
        ret.update({"do_reg_loss_unlabeled": cfg.DOMAIN_ADAPT.LOSSES.LOC_LOSS_ENABLED,
                    "do_quality_loss_weight_unlabeled": cfg.DOMAIN_ADAPT.LOSSES.QUALITY_LOSS_WEIGHT_ENABLED})

        # domain alighment modules
        if cfg.MODEL.SADA.ENABLED:
            ret.update({"sada_heads": SADA(cfg)})

        return ret

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], labeled: Bool = True, do_sada: Bool = False):
        # hack in PT related stuff as needed
        # our Trainer handles the "unsup_data_weak" case as a separate inference step
        if labeled:
            self.roi_heads.branch = "supervised"
            self.proposal_generator.danchor = False
        else:
            self.roi_heads.branch = "unsupervised"
            self.proposal_generator.danchor = True

        # run forward pass as usual
        output = super(DARCNN, self).forward(batched_inputs)

        if self.training:
            # handle any domain alignment modules
            if do_sada:
                assert self.sada_heads is not None, "SADA is enabled but no SADA module is defined"
                da_losses = self.get_sada_losses(labeled)
                for k, v in da_losses.items():
                    output[k] = v

            # handle any loss modifications
            if not labeled:
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

    def get_sada_losses(self, labeled: Bool):
        domain_label = 1 if labeled else 0

        img_features = list(self.backbone_io.output.values())
        device = img_features[0].device
        img_targets = torch.ones(len(img_features), dtype=torch.long, device=device) * domain_label

        proposals = [x.proposal_boxes for x in self.roih_io.output[0]] # roih_out = proposals, losses
        instance_features = self.boxhead_io.output
        instance_targets = torch.ones(sum([len(b) for b in proposals]), dtype=torch.long, device=device) * domain_label

        da_losses = self.sada_heads(img_features, instance_features, instance_targets, proposals, img_targets)
        return da_losses