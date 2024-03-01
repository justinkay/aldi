import torch
from typing import Dict, List

from detectron2.config import configurable
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.layers import cat, cross_entropy

from aldi.align import AlignMixin
from aldi.distill import DistillMixin
from aldi.helpers import SaveIO


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


@META_ARCH_REGISTRY.register()
class ExtraRCNN(ALDI):
    """Extensions needed to reproduce prior work.
    Right now, additions support: MIC (quality loss weight), PT."""

    @configurable
    def __init__(
        self,
        *,
        do_quality_loss_weight_unlabeled: bool = False,
        do_danchor_labeled: bool = False,
        do_danchor_unlabeled: bool = False,
        **kwargs
    ):
        super(ExtraRCNN, self).__init__(**kwargs)

        self.do_quality_loss_weight_unlabeled = do_quality_loss_weight_unlabeled
        self.do_danchor_labeled = do_danchor_labeled
        self.do_danchor_unlabeled = do_danchor_unlabeled

        # register hooks so we can grab output of sub-modules
        self.backbone_io, self.rpn_io, self.roih_io, self.boxhead_io, self.boxpred_io = SaveIO(), SaveIO(), SaveIO(), SaveIO(), SaveIO()
        self.backbone.register_forward_hook(self.backbone_io)
        self.proposal_generator.register_forward_hook(self.rpn_io)
        self.roi_heads.register_forward_hook(self.roih_io)
        self.roi_heads.box_head.register_forward_hook(self.boxhead_io)
        self.roi_heads.box_predictor.register_forward_hook(self.boxpred_io)

    @classmethod
    def from_config(cls, cfg):
        ret = super(ExtraRCNN, cls).from_config(cfg)
        ret.update({"do_quality_loss_weight_unlabeled": cfg.DOMAIN_ADAPT.LOSSES.QUALITY_LOSS_WEIGHT_ENABLED,
                    "do_danchor_labeled": cfg.GRCNN.LEARN_ANCHORS_LABELED,
                    "do_danchor_unlabeled": cfg.GRCNN.LEARN_ANCHORS_UNLABELED,
                    })
        return ret

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], labeled: bool = True, do_align: bool = False):
        # hack in PT related stuff as needed
        # our Trainer handles the "unsup_data_weak" case as a separate inference step
        self.roi_heads.branch = "supervised" if labeled else "unsupervised"
        if self.proposal_generator is not None:
            self.proposal_generator.branch = "supervised" if labeled else "unsupervised"
            self.proposal_generator.danchor = self.do_danchor_labeled if labeled else self.do_danchor_unlabeled

        # run forward pass as usual
        output = super(ExtraRCNN, self).forward(batched_inputs)

        # MIC quality loss weighting
        if self.training and not labeled and self.do_quality_loss_weight_unlabeled:
            proposals, _ = self.roih_io.output
            predictions = self.boxpred_io.output
            scores, _ = predictions
            gt_classes = (cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0))
            quality = torch.max(torch.softmax(scores, dim=1), dim=1)[0]
            loss_cls = torch.mean(quality * cross_entropy(scores, gt_classes, reduction="none"))
            output["loss_cls"] = loss_cls * self.roi_heads.box_predictor.loss_weight.get("loss_cls", 1.0)

        return output

    def inference(self, *args, **kwargs):
        # hack in PT stuff
        self.roi_heads.branch = "supervised"
        if self.proposal_generator is not None:
            self.proposal_generator.branch = "supervised"
        return super(ExtraRCNN, self).inference(*args, **kwargs)