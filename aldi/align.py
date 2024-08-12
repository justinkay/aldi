import torch
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.modeling import GeneralizedRCNN

from aldi.helpers import SaveIO, grad_reverse
from aldi.sada import SADA

class AlignMixin(GeneralizedRCNN):
    """Any modifications to the torch module itself go here and are mixed in in trainer.ALDI"""

    @configurable
    def __init__(
        self,
        *,
        sada_heads: SADA = None,
        img_da_enabled: bool = False,
        img_da_layer: str = None,
        img_da_weight: float = 0.0,
        img_da_input_dim: int = 256,
        img_da_hidden_dims: list = [256,],
        ins_da_enabled: bool = False,
        ins_da_weight: float = 0.0,
        ins_da_input_dim: int = 1024,
        ins_da_hidden_dims: list = [1024,],
        img_da_impl: str = 'ours',
        **kwargs
    ):
        super(AlignMixin, self).__init__(**kwargs)
        self.img_da_layer = img_da_layer
        self.img_da_weight = img_da_weight
        self.ins_da_weight = ins_da_weight

        self.sada_heads = sada_heads
        self.img_align = None
        if img_da_enabled:
            if img_da_impl == 'at':
                self.img_align = ATDiscriminator(img_da_input_dim)
            else:
                self.img_align = ConvDiscriminator(img_da_input_dim, hidden_dims=img_da_hidden_dims) if img_da_enabled else None
        self.ins_align = FCDiscriminator(ins_da_input_dim, hidden_dims=ins_da_hidden_dims) if ins_da_enabled else None

        # register hooks so we can grab output of sub-modules
        self.backbone_io, self.rpn_io, self.roih_io, self.boxhead_io = SaveIO(), SaveIO(), SaveIO(), SaveIO()
        self.backbone.register_forward_hook(self.backbone_io)
        self.proposal_generator.register_forward_hook(self.rpn_io)
        self.roi_heads.register_forward_hook(self.roih_io)

        if ins_da_enabled or sada_heads is not None:
            assert hasattr(self.roi_heads, 'box_head'), "Instance alignment only implemented for ROI Heads with box_head."
            self.roi_heads.box_head.register_forward_hook(self.boxhead_io)

    @classmethod
    def from_config(cls, cfg):
        ret = super(AlignMixin, cls).from_config(cfg)

        if cfg.DOMAIN_ADAPT.ALIGN.SADA_ENABLED:
            ret.update({"sada_heads": SADA(cfg)})

        ret.update({"img_da_enabled": cfg.DOMAIN_ADAPT.ALIGN.IMG_DA_ENABLED,
                    "img_da_layer": cfg.DOMAIN_ADAPT.ALIGN.IMG_DA_LAYER,
                    "img_da_weight": cfg.DOMAIN_ADAPT.ALIGN.IMG_DA_WEIGHT,
                    "img_da_input_dim": cfg.DOMAIN_ADAPT.ALIGN.IMG_DA_INPUT_DIM,
                    "img_da_hidden_dims": cfg.DOMAIN_ADAPT.ALIGN.IMG_DA_HIDDEN_DIMS,
                    "img_da_impl": cfg.DOMAIN_ADAPT.ALIGN.IMG_DA_IMPL,
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
                if self.sada_heads is not None:
                    img_targets = torch.ones(len(img_features), dtype=torch.long, device=device) * domain_label
                    proposals = [x.proposal_boxes for x in self.roih_io.output[0]] # roih_out = proposals, losses
                    instance_features = self.boxhead_io.output
                    instance_targets = torch.ones(sum([len(b) for b in proposals]), dtype=torch.long, device=device) * domain_label
                    output.update(self.sada_heads(img_features, instance_features, instance_targets, proposals, img_targets))
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
            elif self.img_align or self.ins_align or self.sada_heads:
                # need to utilize the modules at some point during the forward pass or PyTorch complains.
                # this is only an issue when cfg.SOLVER.BACKWARD_AT_END=False, because intermediate backward()
                # calls may not have used alignment heads
                # see: https://github.com/pytorch/pytorch/issues/43259#issuecomment-964284292
                fake_output = 0
                for aligner in [self.img_align, self.ins_align, self.sada_heads]:
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

# Discriminator copied from AT codebase: https://github.com/facebookresearch/adaptive_teacher/blob/main/adapteacher/modeling/meta_arch/rcnn.py
class ATDiscriminator(torch.nn.Module):
    def __init__(self, num_classes, ndf1=256, ndf2=128):
        super(ATDiscriminator, self).__init__()

        self.conv1 = torch.nn.Conv2d(num_classes, ndf1, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(ndf1, ndf2, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(ndf2, ndf2, kernel_size=3, padding=1)
        self.classifier = torch.nn.Conv2d(ndf2, 1, kernel_size=3, padding=1)

        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x