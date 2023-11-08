import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from detectron2.config import configurable
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.layers import cat
from detectron2.layers.wrappers import cross_entropy
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.box_regression import _dense_box_regression_loss
from fvcore.nn import smooth_l1_loss

from helpers import SaveIO, ManualSeed, ReplaceProposalsOnce, set_attributes
from pseudolabeler import PseudoLabeler

class Distiller:

    def __init__(self, teacher, student, do_hard_cls=False, do_hard_obj=False, do_hard_rpn_reg=False, do_hard_roi_reg=False,
                 do_cls_dst=False, do_obj_dst=False, do_rpn_reg_dst=False, do_roih_reg_dst=False, do_hint=False,
                 cls_temperature=1.0, obj_temperature=1.0, cls_loss_type="CE", pseudo_label_threshold=0.8, pseudo_label_method="thresholding"):
        set_attributes(self, locals())
        self.register_hooks()
        self.pseudo_labeler = PseudoLabeler(teacher, pseudo_label_threshold, pseudo_label_method)

    @classmethod
    def from_config(cls, teacher, student, cfg):
        return Distiller(teacher, student,
                        do_hard_cls=cfg.DOMAIN_ADAPT.DISTILL.HARD_ROIH_CLS_ENABLED,
                        do_hard_obj=cfg.DOMAIN_ADAPT.DISTILL.HARD_OBJ_ENABLED,
                        do_hard_rpn_reg=cfg.DOMAIN_ADAPT.DISTILL.HARD_RPN_REG_ENABLED,
                        do_hard_roi_reg=cfg.DOMAIN_ADAPT.DISTILL.HARD_ROIH_REG_ENABLED,
                        do_cls_dst=cfg.DOMAIN_ADAPT.DISTILL.ROIH_CLS_ENABLED, 
                        do_obj_dst=cfg.DOMAIN_ADAPT.DISTILL.OBJ_ENABLED,
                        do_rpn_reg_dst=cfg.DOMAIN_ADAPT.DISTILL.RPN_REG_ENABLED,
                        do_roih_reg_dst=cfg.DOMAIN_ADAPT.DISTILL.ROIH_REG_ENABLED,
                        do_hint=cfg.DOMAIN_ADAPT.DISTILL.HINT_ENABLED,
                        cls_temperature=cfg.DOMAIN_ADAPT.DISTILL.CLS_TMP,
                        obj_temperature=cfg.DOMAIN_ADAPT.DISTILL.OBJ_TMP,
                        cls_loss_type=cfg.DOMAIN_ADAPT.CLS_LOSS_TYPE,
                        pseudo_label_threshold=cfg.DOMAIN_ADAPT.TEACHER.THRESHOLD,
                        pseudo_label_method=cfg.DOMAIN_ADAPT.TEACHER.PSEUDO_LABEL_METHOD)

    def register_hooks(self):
        self.student_rpn_io, self.student_rpn_head_io, self.student_boxpred_io = SaveIO(), SaveIO(), SaveIO()
        self.teacher_backbone_io, self.teacher_rpn_head_io, self.teacher_boxpred_io, self.teacher_anchor_io = SaveIO(), SaveIO(), SaveIO(), SaveIO()
        
        student_model = self.student.module if type(self.student) is DDP else self.student
        teacher_model = self.teacher.module if type(self.teacher) is DDP else self.teacher

        student_model.proposal_generator.register_forward_hook(self.student_rpn_io)
        student_model.proposal_generator.rpn_head.register_forward_hook(self.student_rpn_head_io)
        student_model.roi_heads.box_predictor.register_forward_hook(self.student_boxpred_io)

        teacher_model.backbone.register_forward_hook(self.teacher_backbone_io)
        teacher_model.proposal_generator.rpn_head.register_forward_hook(self.teacher_rpn_head_io)
        teacher_model.roi_heads.box_predictor.register_forward_hook(self.teacher_boxpred_io)
        teacher_model.proposal_generator.anchor_generator.register_forward_hook(self.teacher_anchor_io)

        # Make sure seeds are the same for proposal sampling in teacher/student
        self.seeder = ManualSeed()
        teacher_model.roi_heads.register_forward_pre_hook(self.seeder)
        student_model.roi_heads.register_forward_pre_hook(self.seeder)

        # Teacher and student second stage need to have the same input proposals in order to distill predictions on those proposals
        self.teacher_proposal_replacer = ReplaceProposalsOnce()
        teacher_model.roi_heads.register_forward_pre_hook(self.teacher_proposal_replacer)

        if self.do_hint:
            self.student_hint_io = SaveIO()
            student_model.hint_adapter.register_forward_hook(self.student_hint_io)
            self.student_bottom_up_io = SaveIO()
            student_model.backbone.bottom_up.register_forward_hook(self.student_bottom_up_io)
            self.teacher_bottom_up_io = SaveIO()
            teacher_model.backbone.bottom_up.register_forward_hook(self.teacher_bottom_up_io)

    def distill_enabled(self):
        return any([self.do_hard_cls, self.do_hard_obj, self.do_hard_rpn_reg, self.do_hard_roi_reg,
                    self.do_cls_dst, self.do_obj_dst, self.do_rpn_reg_dst, self.do_roih_reg_dst, self.do_hint])

    def _distill_forward(self, teacher_batched_inputs, student_batched_inputs):
        # first, get hard pseudo labels -- this is done in place
        # even if not included in overall loss, we need them for RPN proposal sampling
        # TODO there may be a more efficient way to do the latter if you don't want hard losses
        self.pseudo_labeler(teacher_batched_inputs, student_batched_inputs)
        
        self.seeder.reset_seed()

        # teacher might be in eval mode -- this is important for inputs/outputs aligning
        was_eval = not self.teacher.training
        if was_eval: 
            self.teacher.train()

        standard_losses = self.student(student_batched_inputs)
        student_proposals, _ = self.student_rpn_io.output

        self.teacher_proposal_replacer.set_proposals(student_proposals)
        with torch.no_grad():
            self.teacher(teacher_batched_inputs)
        
        # return to eval mode if necessary
        if was_eval: 
            self.teacher.eval()

        return standard_losses

    def __call__(self, teacher_batched_inputs, student_batched_inputs):
        losses = {}

        # Do a forward pass to get activations, and get hard pseudo-label losses if desired
        hard_losses = self._distill_forward(teacher_batched_inputs, student_batched_inputs)
        loss_to_attr = {
            "loss_cls": self.do_hard_cls,
            "loss_rpn_cls": self.do_hard_obj,
            "loss_rpn_loc": self.do_hard_rpn_reg,
            "loss_box_reg": self.do_hard_roi_reg,
        }
        for k, v in hard_losses.items():
            if loss_to_attr.get(k, False):
                losses[k] = v
            else:
                # Need to add to standard losses so that the optimizer can see it
                losses[k] = v * 0.0

        losses.update(self.get_rpn_losses(teacher_batched_inputs))
        losses.update(self.get_roih_losses())
        losses.update(self.get_hint_losses())

        return losses
    
    def get_rpn_losses(self, teacher_batched_inputs):
        losses = {}
        student_objectness_logits, student_proposal_deltas = self.student_rpn_head_io.output
        teacher_objectness_logits, teacher_proposal_deltas = self.teacher_rpn_head_io.output

        # the RPN samples proposals for loss computation *after* the RPN head
        # so we need to mimic this logic ourselves to match -- it's a bit complicated to reverse engineer
        rpn = (self.teacher.module if type(self.teacher) is DDP else self.teacher).proposal_generator
        pseudo_gt_labels = torch.stack(rpn.label_and_sample_anchors(self.teacher_anchor_io.output, 
                                                                       [i['instances'].to(self.teacher.device) for i in teacher_batched_inputs])[0])
        valid_mask = torch.flatten(pseudo_gt_labels >= 0) # the proposals we'll compute loss for
        fg_mask = pseudo_gt_labels == 1 # proposals matched to a pseudo GT box

        # Postprocessing -- for now just sharpening
        teacher_objectness_probs = torch.sigmoid(cat([torch.flatten(t) for t in teacher_objectness_logits]) / self.obj_temperature)

        # Objectness loss -- compute for all subsampled proposals (use valid_mask)
        if self.do_obj_dst:
            objectness_loss = F.binary_cross_entropy_with_logits(
                cat([torch.flatten(t) for t in student_objectness_logits])[valid_mask],
                teacher_objectness_probs[valid_mask],
                reduction="mean"
            )
            losses["loss_obj_bce"] = objectness_loss

        # Regression loss -- compute only for positive proposals (use fg_mask)
        if self.do_rpn_reg_dst:
            fg_mask = torch.repeat_interleave(fg_mask, repeats=4)
            loss_rpn_reg = smooth_l1_loss(
                cat([torch.flatten(t) for t in student_proposal_deltas])[fg_mask],
                cat([torch.flatten(t) for t in teacher_proposal_deltas])[fg_mask],
                beta=0.0, # default
                reduction="mean"
            )
            losses["loss_rpn_l1"] = loss_rpn_reg

        return losses
    
    def get_roih_losses(self):
        losses = {}
        student_cls_logits, student_proposal_deltas = self.student_boxpred_io.output
        teacher_cls_logits, teacher_proposal_deltas = self.teacher_boxpred_io.output

        # Postprocessing -- for now just sharpening
        teacher_cls_probs = F.softmax(teacher_cls_logits / self.cls_temperature, dim=1)

        # ROI heads classification loss
        if self.do_cls_dst:
            if self.cls_loss_type == "CE":
                cls_dst_loss = cross_entropy(student_cls_logits, teacher_cls_probs)
            elif self.cls_loss_type == "KL":
                cls_dst_loss = F.kl_div(F.log_softmax(student_cls_logits, dim=1),
                                    F.log_softmax(teacher_cls_logits / self.cls_temperature, dim=1),
                                    reduction="batchmean",
                                    log_target=True)
            else:
                raise ValueError("cls_loss_type must be one of {CE, KL}")
            losses["loss_cls_ce"] = cls_dst_loss

        # ROI box loss
        if self.do_roih_reg_dst:
            # get the regression targets for all pseudo-foreground proposals
            bg_idx = teacher_cls_logits.shape[1] - 1
            fg_cls = torch.argmax(teacher_cls_logits, dim=1)
            fg_mask = fg_cls != bg_idx
            
            fg_teacher_deltas = teacher_proposal_deltas.view(-1, bg_idx, 4)[
                fg_mask, fg_cls[fg_mask], :
            ]
            fg_student_deltas = student_proposal_deltas.view(-1, bg_idx, 4)[
                fg_mask, fg_cls[fg_mask], :
            ]

            loss_roih_reg = smooth_l1_loss(
                    fg_student_deltas,
                    fg_teacher_deltas,
                    beta=0.0, # default
                    reduction="sum"
                )
            
            # normalize by the total number of regions so that each proposal is given
            # equal weight; see detectron2.modeling.roi_heads.fast_rcnn.py:box_reg_loss
            normalizer = teacher_cls_logits.shape[0]
            losses["loss_roih_l1"] = loss_roih_reg / normalizer

        return losses
    
    def get_hint_losses(self):
        losses = {}
        # Assumes DistillMixin has been used
        if self.do_hint:
            student_model = self.student.module if type(self.student) is DDP else self.student
            # teacher_features = [self.teacher_backbone_io.output[f] for f in student_model.hint_adapter.in_features]
            teacher_features = [self.teacher_bottom_up_io.output[f] for f in student_model.hint_adapter.in_features]
            hint_loss = 0.0
            for student_feat, teacher_feat in zip(self.student_hint_io.output, teacher_features):
                hint_loss += F.mse_loss(student_feat, teacher_feat, reduction="mean")
            losses["loss_hint_l2"] = 0.5 * hint_loss / len(teacher_features)
        return losses


class DistillMixin(GeneralizedRCNN):
    """Any modifications to the torch module itself go here and are mixed in in rcnn.ALDI"""

    class HintAdaptLayer(torch.nn.Module):
        def __init__(self, hint_channels=512, hint_layer="res3"):
            super().__init__()
            self.hint_adapter = torch.nn.Conv2d(in_channels=hint_channels, out_channels=hint_channels, kernel_size=1)
            # initialize to identity matrix for initial training stability
            with torch.no_grad():
                self.hint_adapter.weight.zero_()
                for i in range(hint_channels):
                    self.hint_adapter.weight[i, i, 0, 0] = 1
                self.hint_adapter.bias.zero_()
            self.in_features = [hint_layer]

        def forward(self, x):
            """Handles multi-level features."""
            in_features = [x[f] for f in self.in_features]
            out_features = []
            for f in in_features:
                out_features.append(self.hint_adapter(f))
            return out_features

    @configurable
    def __init__(self, *, do_hint=False, hint_channels=512, hint_layer="res3", **kwargs):
        super(DistillMixin, self).__init__(**kwargs)
        self.do_hint = do_hint
        if do_hint:
            self.backbone_io = SaveIO()
            self.backbone.bottom_up.register_forward_hook(self.backbone_io)
            self.hint_adapter = DistillMixin.HintAdaptLayer(hint_channels=hint_channels, hint_layer=hint_layer)

    @classmethod
    def from_config(cls, cfg):
        ret = super(DistillMixin, cls).from_config(cfg)
        ret.update({"do_hint": cfg.DOMAIN_ADAPT.DISTILL.HINT_ENABLED,
                    "hint_channels": cfg.DOMAIN_ADAPT.DISTILL.HINT_CHANNELS,
                    "hint_layer": cfg.DOMAIN_ADAPT.DISTILL.HINT_LAYER
                    })
        return ret

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        if self.do_hint and self.training:
            self.hint_adapter(self.backbone_io.output)
            # don't compute losses here; but make sure parameters are seen as being used
            # this is needed for any forward passes that don't use the hint adapter
            output["_"] = sum([p.sum() for p in self.hint_adapter.parameters()]) * 0
        return output
