from collections import OrderedDict

import torch
from torch import nn

class EMATeacher(nn.Module):
    """
    EMA teacher model for mean teacher training
    Implementation from: https://github.com/lhoyer/MIC/blob/master/det/maskrcnn_benchmark/modeling/teacher.py
    """

    def __init__(self, model, alpha):
        super(EMATeacher, self).__init__()
        self.ema_model = model
        self.alpha = alpha

    def _init_ema_weights(self, model):
        self.ema_model.load_state_dict(model.state_dict())

    def _update_ema(self, model, iter):
        student_model_dict = model.state_dict()
        new_teacher_dict = OrderedDict()
        for key, value in self.ema_model.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - self.alpha) + value * self.alpha
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.ema_model.load_state_dict(new_teacher_dict)

    def update_weights(self, model, iter):
        # Init/update ema model
        if iter == 0:
            self._init_ema_weights(model)
        if iter > 0:
            self._update_ema(model, iter)

    @torch.no_grad()
    def forward(self, target_img):
        # Generate pseudo-label
        for m in self.ema_model.modules():
            # JK: do we need this?
            m.training = False
        logits = self.ema_model(target_img)
        return logits