from collections import OrderedDict

import torch
from torch import nn

class SaveIO:
    """Simple PyTorch hook to save the output of a nn.module."""
    def __init__(self):
        self.input = None
        self.output = None
        
    def __call__(self, module, module_in, module_out):
        self.input = module_in
        self.output = module_out

class EmaRCNN(nn.Module):
    """EMA of an R-CNN."""
    def __init__(self, model, alpha):
        super(EmaRCNN, self).__init__()
        self.model = model
        self.alpha = alpha
    
        # check it's an R-CNN
        assert hasattr(self.model, "proposal_generator"), "Only R-CNNs are supported by EmaRCNN"
        assert hasattr(self.model, "roi_heads"), "Only R-CNNs are supported by EmaRCNN"

        # register hooks so we can grab output of sub-modules
        self.rpn_io, self.roih_io = SaveIO(), SaveIO()
        self.model.proposal_generator.register_forward_hook(self.rpn_io)
        self.model.roi_heads.register_forward_hook(self.roih_io)

    def _init_ema_weights(self, model):
        self.model.load_state_dict(model.state_dict())

    def _update_ema(self, model, iter):
        student_model_dict = model.state_dict()
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
    def forward(self, target_img):
        """
        Returns:
            RPN outputs (proposals), 
            ROI Heads outputs (pred_instances)
            Full model output (postprocessed_output)
        """
        self.model.eval()
        postprocessed_output = self.model(target_img)   # transformed back into original input space with GeneralizedRCNN._postprocess
        proposals, _ = self.rpn_io.output               # not transformed back to original input space
        pred_instances, _ = self.roih_io.output         # not transformed back to original input space

        return proposals, pred_instances, postprocessed_output