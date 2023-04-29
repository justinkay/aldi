from collections import OrderedDict
import copy 
import torch
from torch import nn

import detectron2.utils.comm as comm


class EMA(nn.Module):
    def __init__(self, model, alpha):
        super(EMA, self).__init__()
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