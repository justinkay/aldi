import copy
import torch.nn.functional as F

from helpers import SaveIO


class Distiller:

    def __init__(self, teacher, student):
        self.teacher = teacher
        self.student = student
        self.register_hooks()

    def register_hooks(self):
        self.teacher_io, self.teacher_backbone_io, self.teacher_rpn_io, self.teacher_roih_io, self.teacher_boxhead_io, self.teacher_boxpred_io = SaveIO(), SaveIO(), SaveIO(), SaveIO(), SaveIO(), SaveIO()
        self.student_io, self.student_backbone_io, self.student_rpn_io, self.student_roih_io, self.student_boxhead_io, self.student_boxpred_io = SaveIO(), SaveIO(), SaveIO(), SaveIO(), SaveIO(), SaveIO()
        
        self.student.register_forward_hook(self.student_io)
        self.student.backbone.register_forward_hook(self.student_backbone_io)
        self.student.proposal_generator.register_forward_hook(self.student_rpn_io)
        self.student.roi_heads.register_forward_hook(self.student_roih_io)
        self.student.roi_heads.box_head.register_forward_hook(self.student_boxhead_io)
        self.student.roi_heads.box_predictor.register_forward_hook(self.student_boxpred_io)

        self.teacher.register_forward_hook(self.teacher_io)
        self.teacher.backbone.register_forward_hook(self.teacher_backbone_io)
        self.teacher.proposal_generator.register_forward_hook(self.teacher_rpn_io)
        self.teacher.roi_heads.register_forward_hook(self.teacher_roih_io)
        self.teacher.roi_heads.box_head.register_forward_hook(self.teacher_boxhead_io)
        self.teacher.roi_heads.box_predictor.register_forward_hook(self.teacher_boxpred_io)

    def __call__(self): #, data):
        # first let's just see if we can hook into the right outputs
        # get the class logits from each:
        # TODO do we need to deepcopy?
        student_cls_scores, student_proposal_deltas = self.student_boxpred_io.output
        teacher_cls_scores, teacher_proposal_deltas = self.teacher_boxpred_io.output

        # student RPN proposals
        student_proposals = self.student_rpn_io.output[0]
        print("student proposals", len(student_proposals)) 
        print(student_proposals[0])

        # make a copy of the teacher second stage
        teacher_copy = copy.deepcopy(self.teacher)
        teacher_copy.proposal_generator = None
        # add student proposals to input and get teacher predictions again
        print("teacher input", self.teacher_io.input)

        # problem 1: train batch size is 1024, inference batch size is 2000
        print(student_cls_scores.shape, teacher_cls_scores.shape)
        print(student_proposal_deltas.shape, teacher_proposal_deltas.shape)
        print(student_cls_scores[0], teacher_cls_scores[0])
        
        # HACK
        teacher_cls_scores = teacher_cls_scores[:student_cls_scores.shape[0]]
        teacher_proposal_deltas = teacher_proposal_deltas[:student_proposal_deltas.shape[0]]

        temperature = 1.0
        teacher_cls_probs = F.softmax(teacher_cls_scores / temperature, dim=1)
        print(teacher_cls_probs[0])
        print(sum(teacher_cls_probs[0]))
        cls_dst_loss = F.cross_entropy(student_cls_scores, teacher_cls_probs)
        # cls_dist_loss = torch.nn.functional.kl_div(student_cls_scores, teacher_cls_probs, reduction='batchmean')
        print(cls_dst_loss)
        

        # proposal deltas shape: num_bbox_reg_classes * box_dim
        # print(student_logits.shape, teacher_logits.shape)