import random
import torch
import torch.nn.functional as F

from detectron2.structures import Instances
from detectron2.layers import cat
from detectron2.layers.wrappers import cross_entropy

from helpers import SaveIO


def distill_forward(teacher, student, teacher_data, student_data):
    """
    Reproducing GeneralizedRCNN.forward in a way that allows us to
    use the outputs of the teacher and student networks in the
    distillation loss.

    In addition to just getting the distillation targets, we have to 
    account for randomness in the proposal sampling process. This is unfortunately
    pretty hard to extract from Detectron2, so we have to reproduce all the forward()
    logic here.
    """
    # teacher might be in eval mode
    was_eval = not teacher.training
    if was_eval: teacher.train()

    teacher_images = teacher.preprocess_image(teacher_data)
    student_images = student.preprocess_image(student_data)
    teacher_gt_instances = [x["instances"].to(teacher.device) for x in teacher_data] # assumes this is pseudolabeled already
    student_gt_instances = [x["instances"].to(student.device) for x in student_data] # assumes this is pseudolabeled already

    # do backbone forward
    teacher_features = teacher.backbone(teacher_images.tensor)
    student_features = student.backbone(student_images.tensor)

    # do RPN forward
    # reset RNG so that proposal sampling is the same for each, so we can directly compare
    rpn_seed = random.randint(0, 2**32 - 1)
    torch.manual_seed(rpn_seed)
    teacher_proposals, teacher_proposal_losses = teacher.proposal_generator(teacher_images, teacher_features, teacher_gt_instances)
    torch.manual_seed(rpn_seed)
    student_proposals, student_proposal_losses = student.proposal_generator(student_images, student_features, student_gt_instances)

    # do ROI heads forward
    # reset RNG; and *also initialize teacher 2nd stage w/ student proposals*
    roih_seed = random.randint(0, 2**32 - 1)
    torch.manual_seed(roih_seed)
    teacher_proposals, teacher_detector_losses = teacher.roi_heads(teacher_images, teacher_features, student_proposals, teacher_gt_instances)
    torch.manual_seed(roih_seed)
    student_proposals, student_detector_losses = student.roi_heads(student_images, student_features, student_proposals, student_gt_instances)

    # return to eval mode if necessary
    if was_eval: teacher.eval()


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
        distill_forward(self.teacher, self.student, teacher_batched_inputs, student_batched_inputs)
        
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
        objectness_loss = F.binary_cross_entropy_with_logits(
            cat([torch.flatten(t) for t in student_objectness_logits]),
            teacher_objectness_probs,
            reduction="mean"
        )

        # RPN box loss
        # TODO

        # ROI heads classification loss
        cls_dst_loss = cross_entropy(student_cls_logits, teacher_cls_probs)
        
        # ROI box loss
        # TODO

        # Feature losses/hints
        # TODO

        return {
            "loss_cls_ce": cls_dst_loss,
            # "loss_obj_bce": objectness_loss,
        }