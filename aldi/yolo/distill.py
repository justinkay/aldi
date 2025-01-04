import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from detectron2.layers.wrappers import cross_entropy

from aldi.distill import DISTILLER_REGISTRY, DISTILL_MIXIN_REGISTRY, Distiller
from aldi.helpers import SaveIO, set_attributes
from aldi.pseudolabeler import PseudoLabeler

from .libs.Yolo_Detectron2.yolo_detectron2 import Yolo


@DISTILLER_REGISTRY.register()
class YoloDistiller(Distiller):

    def __init__(self, teacher, student, do_hard_cls=False, do_hard_obj=False, do_hard_rpn_reg=False, do_hard_roi_reg=False,
                 do_cls_dst=False, do_obj_dst=False, do_rpn_reg_dst=False, do_roih_reg_dst=False,
                 cls_temperature=1.0, obj_temperature=1.0, cls_loss_type="CE", pseudo_label_threshold=0.8):
        assert not do_hard_rpn_reg, "enabling DOMAIN_ADAPT.DISTILL.HARD_RPN_REG_ENABLED is not supported for Yolo"
        set_attributes(self, locals())
        self.register_hooks()
        self.pseudo_labeler = PseudoLabeler(teacher, pseudo_label_threshold)

    @classmethod
    def from_config(cls, cfg, teacher, student):
        return YoloDistiller(teacher, student,
                        do_hard_cls=cfg.DOMAIN_ADAPT.DISTILL.HARD_ROIH_CLS_ENABLED,
                        do_hard_obj=cfg.DOMAIN_ADAPT.DISTILL.HARD_OBJ_ENABLED,
                        do_hard_rpn_reg=cfg.DOMAIN_ADAPT.DISTILL.HARD_RPN_REG_ENABLED,
                        do_hard_roi_reg=cfg.DOMAIN_ADAPT.DISTILL.HARD_ROIH_REG_ENABLED,
                        do_cls_dst=cfg.DOMAIN_ADAPT.DISTILL.ROIH_CLS_ENABLED, 
                        do_obj_dst=cfg.DOMAIN_ADAPT.DISTILL.OBJ_ENABLED,
                        do_rpn_reg_dst=cfg.DOMAIN_ADAPT.DISTILL.RPN_REG_ENABLED,
                        do_roih_reg_dst=cfg.DOMAIN_ADAPT.DISTILL.ROIH_REG_ENABLED,
                        cls_temperature=cfg.DOMAIN_ADAPT.DISTILL.CLS_TMP,
                        obj_temperature=cfg.DOMAIN_ADAPT.DISTILL.OBJ_TMP,
                        cls_loss_type=cfg.DOMAIN_ADAPT.CLS_LOSS_TYPE,
                        pseudo_label_threshold=cfg.DOMAIN_ADAPT.TEACHER.THRESHOLD)

    def register_hooks(self):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            x (list[Tensor]): #nl tensors,
                                each having shape [N, na, Hi, Wi, nc + 5]
            z (Tensor) : [N, nl*na*(sum of grid sizes) , no] indictaing
                    1. Box position z[..., 0:2]
                    2. Box width and height z[..., 2:4]
                    3. Objectness z[..., 5]
                    4. Class probabilities z[..., 6:]
        """
        self.student_head_io = SaveIO()
        self.teacher_head_io = SaveIO()

        student_model = self.student.module if type(self.student) is DDP else self.student
        teacher_model = self.teacher.module if type(self.teacher) is DDP else self.teacher

        student_model.model[-1].register_forward_hook(self.student_head_io)
        teacher_model.model[-1].register_forward_hook(self.teacher_head_io)

    def _distill_forward(self, teacher_batched_inputs, student_batched_inputs):
        # first, get hard pseudo labels -- this is done in place
        self.pseudo_labeler(teacher_batched_inputs, student_batched_inputs)

        # teacher might be in eval mode -- this is important for inputs/outputs aligning
        was_eval = not self.teacher.training
        if was_eval: 
            self.teacher.train()

        standard_losses = self.student(student_batched_inputs)

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
            "loss_obj": self.do_hard_obj,
            "loss_box": self.do_hard_roi_reg,
        }
        for k, v in hard_losses.items():
            if loss_to_attr.get(k, False):
                losses[k] = v
            else:
                # Need to add to standard losses so that the optimizer can see it
                losses[k] = v * 0.0

        losses.update(self.get_yolo_soft_losses(student_batched_inputs))
        if self.do_roih_reg_dst:
            # soft reg = hard reg
            losses.update({"loss_soft_reg": hard_losses["loss_box"]}) 

        return losses
    
    def get_yolo_soft_losses(self, student_batched_inputs):
        student_logits = self.student_head_io.output
        teacher_logits = self.teacher_head_io.output

        if self.do_cls_dst:
            # we compute classification and regresion loss only for pseudo-foreground objects
            # so first match student predictions to high-scoring teacher predictions
            psuedo_instances = [x["instances"].to(self.student.device) for x in student_batched_inputs]
            student_images = self.student.preprocess_image(student_batched_inputs)
            tcls, tbox, indices, anchors = self.student.loss.build_targets(
                student_logits, psuedo_instances, student_images) 

        lcls, lobj = torch.zeros(1, device=self.student.device), torch.zeros(
            1, device=self.student.device)

        # Losses
        for i, (pi, ti) in enumerate(zip(student_logits, teacher_logits)):  # layer index, layer predictions
            if self.do_cls_dst:
                b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
                n = b.shape[0]  # number of targets
                if n:
                    # prediction subset corresponding to targets
                    ps = pi[b, a, gj, gi]
                    ts = ti[b, a, gj, gi]

                    # Classification
                    if self.student.loss.nc > 1:  # cls loss (only if multiple classes)
                        teacher_cls_probs = F.softmax(ts[:, 5:] / self.cls_temperature, dim=-1)
                        lcls += cross_entropy(ps[:, 5:], teacher_cls_probs)

            if self.do_obj_dst:
                teacher_objectness_probs = torch.sigmoid(ti[..., 4] / self.obj_temperature)
                objectness_loss = F.binary_cross_entropy_with_logits(pi[..., 4], 
                                                teacher_objectness_probs, reduction="mean")
                lobj += objectness_loss * self.student.loss.balance[i]  # obj loss

        lobj *= self.student.loss.obj_loss_gain
        lcls *= self.student.loss.cls_loss_gain

        return {
            "loss_soft_obj": lobj,
            "loss_soft_cls": lcls,
        }

# Any modifications to the torch module itself go here and are mixed in
# See align.py for an example
# For now, no modifications are needed
@DISTILL_MIXIN_REGISTRY.register()
class YoloDistillMixin(Yolo): pass