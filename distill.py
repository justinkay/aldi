import copy
import random
import torch
import torch.nn.functional as F

from detectron2.structures import Instances
from detectron2.layers import cat
from detectron2.layers.wrappers import cross_entropy

from helpers import SaveIO, ManualSeed, ReplaceProposalsOnce


class Distiller:

    def __init__(self, teacher, student, do_cls_dst=False, do_obj_dst=False, do_rpn_reg_dst=False, do_roi_reg_dst=False, do_hint=False):
        self.teacher = teacher
        self.student = student
        self.do_cls_dst = do_cls_dst
        self.do_obj_dst = do_obj_dst
        self.do_rpn_reg_dst = do_rpn_reg_dst
        self.do_roi_reg_dst = do_roi_reg_dst
        self.do_hint = do_hint

        self.register_hooks()

        # TODO
        self.obj_temperature = 1.0
        self.cls_temperature = 1.0

    def register_hooks(self):
        self.teacher_io, self.teacher_backbone_io, self.teacher_rpn_io, self.teacher_rpn_head_io, self.teacher_roih_io, self.teacher_boxhead_io, self.teacher_boxpred_io = SaveIO(), SaveIO(), SaveIO(), SaveIO(), SaveIO(), SaveIO(), SaveIO()
        self.student_io, self.student_backbone_io, self.student_rpn_io, self.student_rpn_head_io, self.student_roih_io, self.student_boxhead_io, self.student_boxpred_io = SaveIO(), SaveIO(), SaveIO(), SaveIO(), SaveIO(), SaveIO(), SaveIO()
        
        student_model = self.student.module if type(self.student) is torch.nn.parallel.DistributedDataParallel else self.student
        teacher_model = self.teacher.module if type(self.teacher) is torch.nn.parallel.DistributedDataParallel else self.teacher

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

        self.seeder = ManualSeed()
        teacher_model.proposal_generator.register_forward_pre_hook(self.seeder)
        student_model.proposal_generator.register_forward_pre_hook(self.seeder)
        teacher_model.roi_heads.register_forward_pre_hook(self.seeder)
        student_model.roi_heads.register_forward_pre_hook(self.seeder)

        self.teacher_proposal_replacer = ReplaceProposalsOnce()
        teacher_model.roi_heads.register_forward_pre_hook(self.teacher_proposal_replacer)

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
        
        # TODO these are actually the hard pseudo-label losses
        standard_losses = self._distill_forward(teacher_batched_inputs, student_batched_inputs)
        
        # get student outputs
        student_cls_logits, student_proposal_deltas = self.student_boxpred_io.output

        # first stage student outputs
        student_objectness_logits, student_proposal_deltas = self.student_rpn_head_io.output
        teacher_objectness_logits, teacher_proposal_deltas = self.teacher_rpn_head_io.output
        
        # get second-stage teacher outputs using student proposals
        teacher_cls_logits, teacher_proposal_deltas = self.teacher_boxpred_io.output

        # Postprocessing -- for now just sharpening
        teacher_objectness_probs = torch.sigmoid(cat([torch.flatten(t) for t in teacher_objectness_logits]) / self.obj_temperature)
        teacher_cls_probs = F.softmax(teacher_cls_logits / self.cls_temperature, dim=1)

        # RPN objectness loss
        if self.do_obj_dst:
            objectness_loss = F.binary_cross_entropy_with_logits(
                cat([torch.flatten(t) for t in student_objectness_logits]),
                teacher_objectness_probs,
                reduction="mean"
            )
            losses["loss_obj_bce"] = objectness_loss

        # RPN box loss
        # TODO

        # ROI heads classification loss
        if self.do_cls_dst:
            cls_dst_loss = cross_entropy(student_cls_logits, teacher_cls_probs)
            losses["loss_cls_ce"] = cls_dst_loss

            # cls_dst_loss = F.kl_div(F.log_softmax(student_cls_logits, dim=1),
            #                         F.log_softmax(teacher_cls_logits / self.cls_temperature, dim=1),
            #                         reduction="batchmean",
            #                         log_target=True)
            # losses["loss_cls_kl"] = cls_dst_loss

        # ROI box loss
        # TODO

        # Feature losses/hints
        # TODO

        # if DEBUG:
            # self.visualize_batch()

        # hacky
        # TODO add each one to their respective loss entry in standard_losses
        for k, v in standard_losses.items():
            losses["loss_cls_ce"] += v * 0.0

        return losses

    def visualize_batch(self):
        import matplotlib.pyplot as plt
        import numpy as np
        import time

        student_cls_logits, _ = self.student_boxpred_io.output
        teacher_cls_logits, _ = self.teacher_boxpred_io.output
        teacher_cls_probs = F.softmax(teacher_cls_logits / self.cls_temperature, dim=1)

        # Average logits across the batch for each class
        student_avg_logits = student_cls_logits.mean(dim=0).detach().cpu().numpy()
        teacher_avg_logits = teacher_cls_logits.mean(dim=0).detach().cpu().numpy()

        # Compute the differences
        differences = student_avg_logits - teacher_avg_logits

        # Plotting
        fig, axes = plt.subplots(2, 1, figsize=(10, 12))

        # Plot average logits per class for both student and teacher
        axes[0].bar(np.arange(student_avg_logits.shape[0]), student_avg_logits, alpha=0.7, label='Student')
        axes[0].bar(np.arange(teacher_avg_logits.shape[0]), teacher_avg_logits, alpha=0.5, label='Teacher')
        axes[0].set_title('Average Logits per Class')
        axes[0].set_ylabel('Logit Value')
        axes[0].set_xlabel('Class Index')
        axes[0].legend()

        # Plot differences
        axes[1].bar(np.arange(differences.shape[0]), differences, color='r', alpha=0.7)
        axes[1].set_title('Difference in Average Logits per Class (Student - Teacher)')
        axes[1].set_ylabel('Difference')
        axes[1].set_xlabel('Class Index')

        # Save the plot
        timestamp = int(time.time())  # Unix timestamp
        plt.tight_layout()
        plt.savefig(f"debug_{timestamp}.png")