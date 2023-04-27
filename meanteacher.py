from collections import OrderedDict
import copy 
import torch
from torch import nn

import detectron2.utils.comm as comm


class EmaRCNN(nn.Module):
    """EMA of an R-CNN."""
    def __init__(self, model, alpha):
        super(EmaRCNN, self).__init__()
        self.model = copy.deepcopy(model)
        self.alpha = alpha

    def _get_student_dict(self, model):
        # account for DDP
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in model.state_dict().items()
            }
        else:
            student_model_dict = model.state_dict()
        return student_model_dict

    def _init_ema_weights(self, model):
        self.model.load_state_dict(self._get_student_dict(model))

    def _update_ema(self, model, iter):
        print("updating", iter)
        student_model_dict = self._get_student_dict(model)

        # update teacher
        new_teacher_dict = OrderedDict()
        for key, value in self.model.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - self.alpha) + value * self.alpha
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model.load_state_dict(new_teacher_dict)

    def update_weights(self, model, iter):
        # Init/update ema model
        if iter == 0:
            self._init_ema_weights(model)
        if iter > 0:
            self._update_ema(model, iter)

    @torch.no_grad()
    def forward(self, target_img, do_postprocess=True):
        """
        Method is not really used because the intermediate hooks don't help us much.
            (Because it turns out the ROI heads outputs are modified inplace by GeneralizedRCNN._postprocess)
        Args:
            do_postprocess: transform outputs back into input space (see GeneralizedRCNN._postprocess)
                            this should probably be disabled if using intermediate results for training
        Returns:
            RPN outputs (proposals), 
            ROI Heads outputs (pred_instances)
            Full model output (postprocessed_output)
        """
        self.model.eval()
        model_output = self.model.inference(target_img, do_postprocess=do_postprocess)
        
        # not transformed back to original input space
        proposals, _ = self.model.rpn_io.output

        # NOTE: these are modified inplace by GeneralizedRCNN._postprocess
        # so keeping this copy isn't doing much for us 
        pred_instances, _ = self.model.roih_io.output

        return proposals, pred_instances, model_output