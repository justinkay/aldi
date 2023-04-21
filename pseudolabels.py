import copy

from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances

def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
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
        valid_map = proposal_bbox_inst.scores > thres

        # create instances containing boxes and gt_classes
        image_shape = proposal_bbox_inst.image_size
        new_proposal_inst = Instances(image_shape)

        # create box
        new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
        new_boxes = Boxes(new_bbox_loc)

        # add boxes to instances
        new_proposal_inst.gt_boxes = new_boxes
        new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
        new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

    return new_proposal_inst

def process_pseudo_label(
    self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
):
    list_instances = []
    num_proposal_output = 0.0
    for proposal_bbox_inst in proposals_rpn_unsup_k:
        # thresholding
        if psedo_label_method == "thresholding":
            proposal_bbox_inst = self.threshold_bbox(
                proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
            )
        else:
            raise ValueError("Unkown pseudo label boxes methods")
        num_proposal_output += len(proposal_bbox_inst)
        list_instances.append(proposal_bbox_inst)
    num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
    return list_instances, num_proposal_output

def remove_label(self, label_data):
    for label_datum in label_data:
        if "instances" in label_datum.keys():
            del label_datum["instances"]
    return label_data

def add_label(self, unlabled_data, label):
    for unlabel_datum, lab_inst in zip(unlabled_data, label):
        unlabel_datum["instances"] = lab_inst
    return unlabled_data

def get_label(self, label_data):
    label_list = []
    for label_datum in label_data:
        if "instances" in label_datum.keys():
            label_list.append(copy.deepcopy(label_datum["instances"]))
    
    return label_list