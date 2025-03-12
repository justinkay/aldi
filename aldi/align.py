import torch
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.modeling import GeneralizedRCNN
from detectron2.utils.registry import Registry

from aldi.helpers import SaveIO, grad_reverse


ALIGN_MIXIN_REGISTRY = Registry("ALIGN_MIXIN")
ALIGN_MIXIN_REGISTRY.__doc__ = """
TODO
"""


@ALIGN_MIXIN_REGISTRY.register()
class AlignMixin(GeneralizedRCNN):
    """Any modifications to the torch module itself go here and are mixed in in trainer.ALDI"""

    @configurable
    def __init__(
        self,
        *,
        img_da_enabled: bool = False,
        img_da_layer: str = None,
        img_da_weight: float = 0.0,
        img_da_input_dim: int = 256,
        img_da_hidden_dims: list = [256,],
        ins_da_enabled: bool = False,
        ins_da_weight: float = 0.0,
        ins_da_input_dim: int = 1024,
        ins_da_hidden_dims: list = [1024,],
        **kwargs
    ):
        super(AlignMixin, self).__init__(**kwargs)
        self.img_da_layer = img_da_layer
        self.img_da_weight = img_da_weight
        self.ins_da_weight = ins_da_weight

        self.img_align = ConvDiscriminator(img_da_input_dim, hidden_dims=img_da_hidden_dims) if img_da_enabled else None
        self.ins_align = FCDiscriminator(ins_da_input_dim, hidden_dims=ins_da_hidden_dims) if ins_da_enabled else None 

        # register hooks so we can grab output of sub-modules
        self.backbone_io, self.rpn_io, self.roih_io, self.boxhead_io = SaveIO(), SaveIO(), SaveIO(), SaveIO()
        self.backbone.register_forward_hook(self.backbone_io)
        self.proposal_generator.register_forward_hook(self.rpn_io)
        self.roi_heads.register_forward_hook(self.roih_io)

        if ins_da_enabled:
            assert hasattr(self.roi_heads, 'box_head'), "Instance alignment only implemented for ROI Heads with box_head."
            self.roi_heads.box_head.register_forward_hook(self.boxhead_io)

    @classmethod
    def from_config(cls, cfg):
        ret = super(AlignMixin, cls).from_config(cfg)

        ret.update({"img_da_enabled": cfg.DOMAIN_ADAPT.ALIGN.IMG_DA_ENABLED,
                    "img_da_layer": cfg.DOMAIN_ADAPT.ALIGN.IMG_DA_LAYER,
                    "img_da_weight": cfg.DOMAIN_ADAPT.ALIGN.IMG_DA_WEIGHT,
                    "img_da_input_dim": cfg.DOMAIN_ADAPT.ALIGN.IMG_DA_INPUT_DIM,
                    "img_da_hidden_dims": cfg.DOMAIN_ADAPT.ALIGN.IMG_DA_HIDDEN_DIMS,
                    "ins_da_enabled": cfg.DOMAIN_ADAPT.ALIGN.INS_DA_ENABLED,
                    "ins_da_weight": cfg.DOMAIN_ADAPT.ALIGN.INS_DA_WEIGHT,
                    "ins_da_input_dim": cfg.DOMAIN_ADAPT.ALIGN.INS_DA_INPUT_DIM,
                    "ins_da_hidden_dims": cfg.DOMAIN_ADAPT.ALIGN.INS_DA_HIDDEN_DIMS,
                    })

        return ret

    def forward(self, *args, do_align=False, labeled=True, **kwargs):
        output = super().forward(*args, **kwargs)
        if self.training:
            if do_align:
                # extract needed info for alignment: domain labels, image features, instance features
                domain_label = 1 if labeled else 0
                img_features = list(self.backbone_io.output.values())
                device = img_features[0].device
                if self.img_align:
                    features = self.backbone_io.output
                    features = grad_reverse(features[self.img_da_layer])
                    domain_preds = self.img_align(features)
                    loss = F.binary_cross_entropy_with_logits(domain_preds, torch.FloatTensor(domain_preds.data.size()).fill_(domain_label).to(device))
                    output["loss_da_img"] = self.img_da_weight * loss
                if self.ins_align:
                    instance_features = self.boxhead_io.output
                    features = grad_reverse(instance_features)
                    domain_preds = self.ins_align(features)
                    loss = F.binary_cross_entropy_with_logits(domain_preds, torch.FloatTensor(domain_preds.data.size()).fill_(domain_label).to(device))
                    output["loss_da_ins"] = self.ins_da_weight * loss
            elif self.img_align or self.ins_align:
                # need to utilize the modules at some point during the forward pass or PyTorch complains.
                # this is only an issue when cfg.SOLVER.BACKWARD_AT_END=False, because intermediate backward()
                # calls may not have used alignment heads
                # see: https://github.com/pytorch/pytorch/issues/43259#issuecomment-964284292
                fake_output = 0
                for aligner in [self.img_align, self.ins_align]:
                    if aligner is not None:
                        fake_output += sum([p.sum() for p in aligner.parameters()]) * 0
                output["_da"] = fake_output
        return output

class ConvDiscriminator(torch.nn.Module):
    """A discriminator that uses conv layers."""
    def __init__(self, input_dim, hidden_dims=[], kernel_size=3):
        super(ConvDiscriminator, self).__init__()
        modules = []
        prev_dim = input_dim
        for i, dim in enumerate(hidden_dims):
            modules.append(torch.nn.Conv2d(prev_dim, dim, kernel_size))
            modules.append(torch.nn.ReLU())
            prev_dim = dim
        modules.append(torch.nn.AdaptiveAvgPool2d(1))
        modules.append(torch.nn.Flatten())
        modules.append(torch.nn.Linear(prev_dim, 1))
        self.model = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)
    
class FCDiscriminator(torch.nn.Module):
    """A discriminator that uses fully connected layers."""
    def __init__(self, input_dim, hidden_dims=[]):
        super(FCDiscriminator, self).__init__()
        modules = []
        modules.append(torch.nn.Flatten())
        prev_dim = input_dim
        for i, dim in enumerate(hidden_dims):
            modules.append(torch.nn.Linear(prev_dim, dim))
            modules.append(torch.nn.ReLU())
            prev_dim = dim
        modules.append(torch.nn.Linear(prev_dim, 1))
        self.model = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)