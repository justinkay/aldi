import torch
import numpy as np

from detectron2.structures.boxes import Boxes
# from detectron2.structures.instances import Instances
from gaussian_rcnn.instances import FreeInstances as Instances # TODO: only when necessary

from aug import RandomEraseTransform

class PseudoLabeler:
    def __init__(self, cfg, model):
        self.model = model
        self.threshold = cfg.DOMAIN_ADAPT.TEACHER.THRESHOLD
        self.method = cfg.DOMAIN_ADAPT.TEACHER.PSEUDO_LABEL_METHOD

    def __call__(self, unlabeled_weak, unlabeled_strong):
        return do_pseudo_label(self.model, unlabeled_weak, unlabeled_strong, self.threshold, self.method)

def do_pseudo_label(model, unlabeled_weak, unlabeled_strong, threshold, method):
    data_to_pseudolabel = unlabeled_strong if unlabeled_strong is not None else unlabeled_weak
    
    with torch.no_grad():
        # get predictions from teacher model on weakly-augmented data
        # do_postprocess=False to disable transforming outputs back into original image space
        was_training = model.training
        model.eval()
        teacher_preds = model.inference(unlabeled_weak, do_postprocess=False)
        if was_training: model.train()

        # TODO maybe not the best place for this
        # TODO also config dependent
        erased_boxes = []
        for d in data_to_pseudolabel:
            erased_in_d = []
            for k in ["erased_0", "erased_1", "erased_2"]:
                if d.get(k, None) is not None:
                    erased_in_d.append(d[k])
            erased_boxes.append(erased_in_d)

        # postprocess pseudo labels (thresholding)
        teacher_preds, _ = process_pseudo_label(teacher_preds, threshold, "roih", method, erased_boxes)
        
        # add pseudo labels back as "ground truth"
        data_to_pseudolabel = add_label(data_to_pseudolabel, teacher_preds)
    
    return data_to_pseudolabel

# Modified from Adaptive Teacher ATeacherTrainer:
# - Add scores_logists and boxes_sigma from PT if available
# - Remove predicted boxes in erased regions if applicable
def process_pseudo_label(proposals, cur_threshold, proposal_type, pseudo_label_method="", erased_regions=None):
    list_instances = []
    num_proposal_output = 0.0
    for img_idx, proposal_bbox_inst in enumerate(proposals):
        # thresholding
        if pseudo_label_method == "thresholding":
            proposal_bbox_inst = process_bbox(
                proposal_bbox_inst,
                thres=cur_threshold, 
                proposal_type=proposal_type, 
                erased_regions=erased_regions[img_idx] if erased_regions is not None else None
            )
        elif pseudo_label_method == "probabilistic":
            proposal_bbox_inst = process_bbox(
                proposal_bbox_inst,
                thres=-1.0, 
                proposal_type=proposal_type,
                erased_regions=erased_regions[img_idx] if erased_regions is not None else None
            )
        else:
            raise NotImplementedError("Pseudo label method {} not implemented".format(pseudo_label_method))
        num_proposal_output += len(proposal_bbox_inst)
        list_instances.append(proposal_bbox_inst)
        
    num_proposal_output = num_proposal_output / len(proposals)
    return list_instances, num_proposal_output

# Modified from Adaptive Teacher ATeacherTrainer threshold_bbox:
# - Compatible with Proababilistic Teacher outputs
# - Put new labels on CPU (for compatibility with Visualizer, e.g.)
# - Removed boxes in erased regions if applicable
def process_bbox(proposal_bbox_inst, thres=0.7, proposal_type="roih", erased_regions=None):
    if proposal_type == "rpn":
        valid_map = proposal_bbox_inst.objectness_logits > thres

        # create instances containing boxes and gt_classes
        image_shape = proposal_bbox_inst.image_size
        new_proposal_inst = Instances(image_shape)

        # create box
        new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
        new_boxes = Boxes(new_bbox_loc)

        # add boxes to instances
        new_proposal_inst.gt_boxes = new_boxes
        new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
            valid_map
        ]
    elif proposal_type == "roih":

        # remove boxes in erased regions
        # TODO hacky
        if erased_regions is not None:
            for i, bbox in enumerate(proposal_bbox_inst.pred_boxes):
                valid = True
                for erased in erased_regions:
                    if not all(RandomEraseTransform.modify_erased_annotation(bbox.cpu(), erased.cpu()) >= 0.0):
                        valid = False
                        break
                if not valid:
                    proposal_bbox_inst.scores[i] = 0

        valid_map = proposal_bbox_inst.scores > thres

        # create instances containing boxes and gt_classes
        image_shape = proposal_bbox_inst.image_size
        new_proposal_inst = Instances(image_shape)

        # create box
        new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
        new_boxes = Boxes(new_bbox_loc)

        # add boxes to instances
        new_proposal_inst.gt_boxes = new_boxes.to("cpu")
        new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map].to("cpu")
        new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map].to("cpu")

        # add probabilistic outputs for gaussian RCNN
        if proposal_bbox_inst.has('scores_logists'):
            new_proposal_inst.scores_logists = proposal_bbox_inst.scores_logists.to("cpu")
        if proposal_bbox_inst.has('boxes_sigma'):
            new_proposal_inst.boxes_sigma = proposal_bbox_inst.boxes_sigma.to("cpu")
    else:
        raise ValueError("Unknown proposal type in threshold_bbox")

    return new_proposal_inst

# From Adaptive Teacher ATeacherTrainer
def add_label(unlabled_data, label):
    for unlabel_datum, lab_inst in zip(unlabled_data, label):
        unlabel_datum["instances"] = lab_inst
    return unlabled_data
