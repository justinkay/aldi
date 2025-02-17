import torch

from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances


class PseudoLabeler:
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    def __call__(self, unlabeled_weak, unlabeled_strong):
        return pseudo_label_inplace(self.model, unlabeled_weak, unlabeled_strong, self.threshold)

def pseudo_label_inplace(model, unlabeled_weak, unlabeled_strong, threshold):
    with torch.no_grad():
        # get predictions from teacher model on weakly-augmented data
        # do_postprocess=False to disable transforming outputs back into original image space
        was_training = model.training
        model.eval()
        teacher_preds = model.inference(unlabeled_weak, do_postprocess=False)
        if was_training: model.train()

        # postprocess pseudo labels (thresholding)
        teacher_preds, _ = process_pseudo_label(teacher_preds, threshold)
        
        # add pseudo labels back as "ground truth"
        add_label(unlabeled_weak, teacher_preds)
        if unlabeled_strong is not None:
            add_label(unlabeled_strong, teacher_preds)

# Modified from Adaptive Teacher ATeacherTrainer:
# - Remove RPN option
def process_pseudo_label(proposals, cur_threshold):
    list_instances = []
    num_proposal_output = 0.0
    for proposal_bbox_inst in proposals:
        proposal_bbox_inst = process_bbox(
            proposal_bbox_inst,
            thres=cur_threshold, 
        )
        num_proposal_output += len(proposal_bbox_inst)
        list_instances.append(proposal_bbox_inst)
        
    num_proposal_output = num_proposal_output / len(proposals)
    return list_instances, num_proposal_output

# Modified from Adaptive Teacher ATeacherTrainer threshold_bbox:
# - Remove RPN option
# - Put new labels on CPU (for compatibility with Visualizer, e.g.)
def process_bbox(proposal_bbox_inst, thres=0.7):
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

    return new_proposal_inst

# From Adaptive Teacher ATeacherTrainer
def add_label(unlabled_data, label):
    for unlabel_datum, lab_inst in zip(unlabled_data, label):
        unlabel_datum["instances"] = lab_inst
    return unlabled_data
