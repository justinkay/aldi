import copy
import torch
import torch.nn.functional as F

from detectron2.structures import Instances
from detectron2.layers import cat

from helpers import SaveIO


class Distiller:

    def __init__(self, teacher, student):
        self.teacher = teacher
        self.student = student
        self.register_hooks()

        # TODO
        self.obj_temperature = 1.0
        self.cls_temperature = 1.0

    def register_hooks(self):
        self.teacher_io, self.teacher_backbone_io, self.teacher_rpn_io, self.teacher_rpn_head_io, self.teacher_roih_io, self.teacher_boxhead_io, self.teacher_boxpred_io = SaveIO(), SaveIO(), SaveIO(), SaveIO(), SaveIO(), SaveIO(), SaveIO()
        self.student_io, self.student_backbone_io, self.student_rpn_io, self.student_rpn_head_io, self.student_roih_io, self.student_boxhead_io, self.student_boxpred_io = SaveIO(), SaveIO(), SaveIO(), SaveIO(), SaveIO(), SaveIO(), SaveIO()
        
        self.student.register_forward_hook(self.student_io)
        self.student.backbone.register_forward_hook(self.student_backbone_io)
        self.student.proposal_generator.register_forward_hook(self.student_rpn_io)
        self.student.proposal_generator.rpn_head.register_forward_hook(self.student_rpn_head_io)
        self.student.roi_heads.register_forward_hook(self.student_roih_io)
        self.student.roi_heads.box_head.register_forward_hook(self.student_boxhead_io)
        self.student.roi_heads.box_predictor.register_forward_hook(self.student_boxpred_io)

        self.teacher.register_forward_hook(self.teacher_io)
        self.teacher.backbone.register_forward_hook(self.teacher_backbone_io)
        self.teacher.proposal_generator.register_forward_hook(self.teacher_rpn_io)
        self.teacher.proposal_generator.rpn_head.register_forward_hook(self.teacher_rpn_head_io)
        self.teacher.roi_heads.register_forward_hook(self.teacher_roih_io)
        self.teacher.roi_heads.box_head.register_forward_hook(self.teacher_boxhead_io)
        self.teacher.roi_heads.box_predictor.register_forward_hook(self.teacher_boxpred_io)

    def __call__(self, teacher_batched_inputs, student_batched_inputs):
        # get student outputs
        self.student(student_batched_inputs)
        student_features = self.student.backbone_io.output
        student_proposals = self.student_rpn_io.output[0]
        student_cls_logits, student_proposal_deltas = self.student_boxpred_io.output

        # get first-stage teacher outputs
        with torch.no_grad():
            was_training = self.teacher.training
            self.teacher.eval()
            teacher_preds = self.teacher.inference(teacher_batched_inputs, do_postprocess=False)
            if was_training: self.teacher.train()
        teacher_proposals = self.teacher_rpn_io.output[0]
        teacher_features = self.teacher.backbone_io.output
        
        student_objectness_logits, student_proposal_deltas = self.student_rpn_head_io.output
        teacher_objectness_logits, teacher_proposal_deltas = self.teacher_rpn_head_io.output
        
        # get second-stage teacher outputs using student proposals
        # has to be in training mode so proposal sizes match
        with torch.no_grad():
            preprocessed_images = self.teacher.preprocess_image(teacher_batched_inputs)
            gt_instances = None
            if "instances" in teacher_batched_inputs[0]:
                gt_instances = [x["instances"].to(self.teacher.device) for x in teacher_batched_inputs]
            was_eval = not self.teacher.training
            self.teacher.train()
            _teacher_preds, _ = self.teacher.roi_heads(preprocessed_images, teacher_features, student_proposals, gt_instances)
            if was_eval: self.teacher.eval()
        teacher_cls_logits, teacher_proposal_deltas = self.teacher_boxpred_io.output

        # Postprocessing -- for now just sharpening
        teacher_objectness_probs = torch.sigmoid(cat([torch.flatten(t) for t in teacher_objectness_logits]) / self.obj_temperature)
        teacher_cls_probs = F.softmax(teacher_cls_logits / self.cls_temperature, dim=1)

        # RPN objectness loss
        # usually the RPN samples anchors based on known ground truth boxes
        # but we don't have that here, so we skip the sampling (?)
        # without the sampling though, the background is going to dominate
        # maybe we can:
        #   - sample *student* proposals based on the teacher's predictions?
        #   - use same sampling for teacher's proposals as targets
        objectness_loss = F.binary_cross_entropy_with_logits(
            cat([torch.flatten(t) for t in student_objectness_logits]),
            teacher_objectness_probs,
            reduction="mean"
        )
        # RPN box loss
        # TODO

        # ROI heads classification loss
        cls_dst_loss = F.cross_entropy(student_cls_logits, teacher_cls_probs)
        
        # ROI box loss
        # TODO

        # Feature losses
        # TODO

        # TODO thresholding and pseudolabeling in combo?

        return cls_dst_loss + objectness_loss # + ...  TODO