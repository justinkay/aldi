import torch
from typing import Dict, List
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.layers import cat, cross_entropy

from helpers import SaveIO
from sada import grad_reverse, SADA, FCDiscriminator_img


@META_ARCH_REGISTRY.register()
class DARCNN(GeneralizedRCNN):

    @configurable
    def __init__(
        self,
        *,
        do_reg_loss_unlabeled: bool = True,
        do_quality_loss_weight_unlabeled: bool = False,
        do_danchor_labeled: bool = False,
        do_danchor_unlabeled: bool = False,
        sada_heads: SADA = None,
        dis_type: str = None,
        dis_loss_weight: float = 0.0,
        **kwargs
    ):
        super(DARCNN, self).__init__(**kwargs)

        # TODO: can we autoset these with locals()?
        self.do_reg_loss_unlabeled = do_reg_loss_unlabeled
        self.do_quality_loss_weight_unlabeled = do_quality_loss_weight_unlabeled
        self.do_danchor_labeled = do_danchor_labeled
        self.do_danchor_unlabeled = do_danchor_unlabeled
        self.sada_heads = sada_heads

        # replace SADA with AT-style domain alignment if enabled
        # TODO this could be cleaner
        self.dis_type = dis_type
        self.dis_loss_weight = dis_loss_weight
        if self.dis_type:
            assert sada_heads is None, "Can't have both SADA heads and DA heads"
            self.sada_heads = FCDiscriminator_img(self.backbone._out_feature_channels[self.dis_type])

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

        ret.update({"do_reg_loss_unlabeled": cfg.DOMAIN_ADAPT.LOSSES.LOC_LOSS_ENABLED,
                    "do_quality_loss_weight_unlabeled": cfg.DOMAIN_ADAPT.LOSSES.QUALITY_LOSS_WEIGHT_ENABLED,
                    "do_danchor_labeled": cfg.GRCNN.LEARN_ANCHORS_LABELED,
                    "do_danchor_unlabeled": cfg.GRCNN.LEARN_ANCHORS_UNLABELED,
                    })

        if cfg.MODEL.SADA.ENABLED:
            ret.update({"sada_heads": SADA(cfg)})

        if cfg.MODEL.DA.ENABLED:
            ret.update({"dis_type": cfg.MODEL.DA.DIS_TYPE,
                        "dis_loss_weight": cfg.MODEL.DA.DIS_LOSS_WEIGHT,
                        })

        return ret

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], labeled: bool = True, do_sada: bool = False):
        # hack in PT related stuff as needed
        # our Trainer handles the "unsup_data_weak" case as a separate inference step
        self.roi_heads.branch = "supervised" if labeled else "unsupervised"
        if self.proposal_generator is not None:
            self.proposal_generator.branch = "supervised" if labeled else "unsupervised"
            self.proposal_generator.danchor = self.do_danchor_labeled if labeled else self.do_danchor_unlabeled

        # run forward pass as usual
        output = super(DARCNN, self).forward(batched_inputs)

        if self.training:
            # handle any domain alignment modules
            if do_sada:
                assert self.sada_heads is not None, "SADA is enabled but no SADA module is defined"
                method = "sada" if type(self.sada_heads) == SADA else "da"
                da_losses = self.get_sada_losses(labeled, method)
                for k, v in da_losses.items():
                    output[k] = v
            elif self.sada_heads is not None:
                # need to utilize sada_heads at some point during the forward pass or PyTorch complains.
                # this is only an issue when cfg.SOLVER.BACKWARD_AT_END=False, because intermediate backward()
                # calls may not have used sada_heads
                # see: https://github.com/pytorch/pytorch/issues/43259#issuecomment-964284292
                output["loss_da_img"] = sum([p.sum() for p in self.sada_heads.parameters()]) * 0

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
    
    def inference(self, *args, **kwargs):
        # hack in PT stuff
        self.roi_heads.branch = "supervised"
        if self.proposal_generator is not None:
            self.proposal_generator.branch = "supervised"
        return super(DARCNN, self).inference(*args, **kwargs)

    def get_sada_losses(self, labeled: bool, method="sada"):
        domain_label = 1 if labeled else 0

        img_features = list(self.backbone_io.output.values())
        device = img_features[0].device
        img_targets = torch.ones(len(img_features), dtype=torch.long, device=device) * domain_label

        proposals = [x.proposal_boxes for x in self.roih_io.output[0]] # roih_out = proposals, losses
        instance_features = self.boxhead_io.output
        instance_targets = torch.ones(sum([len(b) for b in proposals]), dtype=torch.long, device=device) * domain_label

        if method == "sada":
            da_losses = self.sada_heads(img_features, instance_features, instance_targets, proposals, img_targets)
        elif method == "da":
            # Adaptive Teacher style
            features = self.backbone_io.output
            features = grad_reverse(features[self.dis_type])
            D_img_out = self.sada_heads(features)
            loss_D_img = F.binary_cross_entropy_with_logits(D_img_out, torch.FloatTensor(D_img_out.data.size()).fill_(domain_label).to(device))
            da_losses = {"loss_da_img": self.dis_loss_weight * loss_D_img}

        return da_losses