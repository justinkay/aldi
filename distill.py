import copy
import random
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from detectron2.structures import Instances
from detectron2.layers import cat
from detectron2.layers.wrappers import cross_entropy
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.box_regression import _dense_box_regression_loss
from fvcore.nn import smooth_l1_loss

from helpers import SaveIO, ManualSeed, ReplaceProposalsOnce, set_attributes


class Distiller:

    def __init__(self, teacher, student, do_hard_cls=False, do_hard_obj=False, do_hard_rpn_reg=False, do_hard_roi_reg=False,
                 do_cls_dst=False, do_obj_dst=False, do_rpn_reg_dst=False, do_roi_reg_dst=False, do_hint=False,
                 cls_temperature=1.0, obj_temperature=1.0, cls_loss_type="CE"):
        set_attributes(self, locals())
        self.register_hooks()

    def register_hooks(self):
        self.teacher_io, self.teacher_backbone_io, self.teacher_rpn_io, self.teacher_rpn_head_io, self.teacher_roih_io, self.teacher_boxhead_io, self.teacher_boxpred_io = SaveIO(), SaveIO(), SaveIO(), SaveIO(), SaveIO(), SaveIO(), SaveIO()
        self.student_io, self.student_backbone_io, self.student_rpn_io, self.student_rpn_head_io, self.student_roih_io, self.student_boxhead_io, self.student_boxpred_io = SaveIO(), SaveIO(), SaveIO(), SaveIO(), SaveIO(), SaveIO(), SaveIO()
        
        student_model = self.student.module if type(self.student) is DDP else self.student
        teacher_model = self.teacher.module if type(self.teacher) is DDP else self.teacher

        student_model.register_forward_hook(self.student_io)
        student_model.backbone.register_forward_hook(self.student_backbone_io)
        student_model.proposal_generator.register_forward_hook(self.student_rpn_io)
        student_model.proposal_generator.rpn_head.register_forward_hook(self.student_rpn_head_io)
        student_model.roi_heads.register_forward_hook(self.student_roih_io)
        student_model.roi_heads.box_head.register_forward_hook(self.student_boxhead_io)
        student_model.roi_heads.box_predictor.register_forward_hook(self.student_boxpred_io)

        teacher_model.register_forward_hook(self.teacher_io)
        teacher_model.backbone.register_forward_hook(self.teacher_backbone_io)
        teacher_model.proposal_generator.register_forward_hook(self.teacher_rpn_io)
        teacher_model.proposal_generator.rpn_head.register_forward_hook(self.teacher_rpn_head_io)
        teacher_model.roi_heads.register_forward_hook(self.teacher_roih_io)
        teacher_model.roi_heads.box_head.register_forward_hook(self.teacher_boxhead_io)
        teacher_model.roi_heads.box_predictor.register_forward_hook(self.teacher_boxpred_io)

        # Make sure seeds are the same for proposal sampling in teacher/student
        # TODO: Don't think we actually need this for RPN?
        self.seeder = ManualSeed()
        teacher_model.proposal_generator.register_forward_pre_hook(self.seeder)
        student_model.proposal_generator.register_forward_pre_hook(self.seeder)
        teacher_model.roi_heads.register_forward_pre_hook(self.seeder)
        student_model.roi_heads.register_forward_pre_hook(self.seeder)

        self.teacher_proposal_replacer = ReplaceProposalsOnce()
        teacher_model.roi_heads.register_forward_pre_hook(self.teacher_proposal_replacer)

        self.teacher_anchor_io = SaveIO()
        teacher_model.proposal_generator.anchor_generator.register_forward_hook(self.teacher_anchor_io)

    def _distill_forward(self, teacher_batched_inputs, student_batched_inputs):
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
            if loss_to_attr[k]:
                losses[k] = v
            else:
                # Need to add to standard losses so that the optimizer can see it
                losses[k] = v * 0.0

        ### RPN ###
        student_objectness_logits, student_proposal_deltas = self.student_rpn_head_io.output
        teacher_objectness_logits, teacher_proposal_deltas = self.teacher_rpn_head_io.output

        # the RPN samples proposals for loss computation *after* the RPN head
        # so we need to mimic this logic ourselves to match -- it's a bit complicated to reverse engineer
        rpn = (self.teacher.module if type(self.teacher) is DDP else self.teacher).proposal_generator
        pseudo_gt_labels = torch.stack(rpn.label_and_sample_anchors(self.teacher_anchor_io.output, 
                                                                       [i['instances'].to(self.teacher.device) for i in teacher_batched_inputs])[0])
        valid_mask = pseudo_gt_labels >= 0 # the proposals we'll compute loss for
        fg_mask = pseudo_gt_labels == 1 # proposals matched to a pseudo GT box

        # TODO is this right? visualize...
        valid_mask = torch.flatten(valid_mask)
        fg_mask = torch.repeat_interleave(fg_mask, repeats=4)

        # Postprocessing -- for now just sharpening
        teacher_objectness_probs = torch.sigmoid(cat([torch.flatten(t) for t in teacher_objectness_logits]) / self.obj_temperature)

        # Objectness loss -- compute for the subsampled proposals
        if self.do_obj_dst:
            objectness_loss = F.binary_cross_entropy_with_logits(
                cat([torch.flatten(t) for t in student_objectness_logits])[valid_mask],
                teacher_objectness_probs[valid_mask],
                reduction="mean"
            )
            losses["loss_obj_bce"] = objectness_loss

        # Regression loss -- compute only for positive proposals
        if self.do_rpn_reg_dst:
            loss_rpn_reg = smooth_l1_loss(
                cat([torch.flatten(t) for t in student_proposal_deltas])[fg_mask],
                cat([torch.flatten(t) for t in teacher_proposal_deltas])[fg_mask],
                beta=0.0, # default
                reduction="mean"
            )
            losses["loss_rpn_l1"] = loss_rpn_reg

        ### ROI Heads ###
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
        print("ROI:")
        print("student cls logits", student_cls_logits.shape)
        print("teacher cls logits", teacher_cls_logits.shape)
        print("student proposal deltas", student_proposal_deltas.shape)
        print("teacher proposal deltas", teacher_proposal_deltas.shape)

        # fg_mask = ...
        # loss_roih_reg = smooth_l1_loss(
        #         cat([torch.flatten(t) for t in student_proposal_deltas])[fg_mask],
        #         cat([torch.flatten(t) for t in teacher_proposal_deltas])[fg_mask],
        #         beta=0.0, # default
        #         reduction="mean"
        #     )
        # losses["loss_roih_l1"] = loss_roih_reg

        # pseudo_gt_labels = torch.stack(rpn.label_and_sample_anchors(self.teacher_anchor_io.output, 
        #                                                                [i['instances'].to(self.teacher.device) for i in teacher_batched_inputs])[0])
        # valid_mask = pseudo_gt_labels >= 0 # the proposals we'll compute loss for
        # fg_mask = pseudo_gt_labels == 1 # proposals matched to a pseudo GT box

        # Feature losses/hints
        # TODO

        return losses