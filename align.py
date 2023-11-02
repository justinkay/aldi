import torch
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.modeling import GeneralizedRCNN

from helpers import SaveIO
from sada import grad_reverse, SADA, FCDiscriminator_img # TODO move these here?


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
        ins_da_enabled: bool = False,
        ins_da_weight: float = 0.0,
        **kwargs
    ):
        super(AlignMixin, self).__init__(**kwargs)
        self.img_da_layer = img_da_layer
        self.img_da_weight = img_da_weight
        self.ins_da_weight = ins_da_weight

        self.sada_heads = sada_heads
        if img_da_enabled:
            self.img_align = ConvDiscriminator(hidden_dims=[256]) # same as RPN head
        if ins_da_enabled:
            self.ins_align = FCDiscriminator(hidden_dims=[1024,1024]) # same as ROI head

        # register hooks so we can grab output of sub-modules
        self.backbone_io, self.rpn_io, self.roih_io, self.boxhead_io, self.boxpred_io = SaveIO(), SaveIO(), SaveIO(), SaveIO(), SaveIO()
        self.backbone.register_forward_hook(self.backbone_io)
        self.proposal_generator.register_forward_hook(self.rpn_io)
        self.roi_heads.register_forward_hook(self.roih_io)
        self.roi_heads.box_head.register_forward_hook(self.boxhead_io)
        self.roi_heads.box_predictor.register_forward_hook(self.boxpred_io)

    @classmethod
    def from_config(cls, cfg):
        ret = super(AlignMixin, cls).from_config(cfg)

        if cfg.DOMAIN_ADAPT.ALIGN.SADA_ENABLED:
            ret.update({"sada_heads": SADA(cfg)})

        ret.update({"img_da_enabled": cfg.DOMAIN_ADAPT.ALIGN.IMG_DA_ENABLED,
                    "img_da_layer": cfg.DOMAIN_ADAPT.ALIGN.IMG_DA_LAYER,
                    "img_da_weight": cfg.DOMAIN_ADAPT.ALIGN.IMG_DA_WEIGHT,
                    "ins_da_enabled": cfg.DOMAIN_ADAPT.ALIGN.INS_DA_ENABLED,
                    "ins_da_weight": cfg.DOMAIN_ADAPT.ALIGN.INS_DA_WEIGHT,
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
                img_targets = torch.ones(len(img_features), dtype=torch.long, device=device) * domain_label
                proposals = [x.proposal_boxes for x in self.roih_io.output[0]] # roih_out = proposals, losses
                instance_features = self.boxhead_io.output
                instance_targets = torch.ones(sum([len(b) for b in proposals]), dtype=torch.long, device=device) * domain_label

                if self.sada_heads is not None:
                    output.update(self.sada_heads(img_features, instance_features, instance_targets, proposals, img_targets))
                if self.img_da_enabled:
                    features = self.backbone_io.output
                    features = grad_reverse(features[self.img_da_layer])
                    domain_preds = self.img_align(features)
                    loss = F.binary_cross_entropy_with_logits(domain_preds, torch.FloatTensor(domain_preds.data.size()).fill_(domain_label).to(device))
                    output["loss_da_img"] = self.img_da_weight * loss
                if self.ins_da_enabled:
                    domain_preds = self.ins_align(instance_features)
                    loss = F.binary_cross_entropy_with_logits(domain_preds, torch.FloatTensor(domain_preds.data.size()).fill_(domain_label).to(device))
                    output["loss_da_ins"] = self.ins_da_weight * loss
                
            elif len(self.aligners) > 0:
                # need to utilize the modules at some point during the forward pass or PyTorch complains.
                # this is only an issue when cfg.SOLVER.BACKWARD_AT_END=False, because intermediate backward()
                # calls may not have used sada_heads
                # see: https://github.com/pytorch/pytorch/issues/43259#issuecomment-964284292
                fake_output = 0
                for aligner in self.aligners:
                    fake_output += sum([p.sum() for p in aligner.parameters()]) * 0

        return output

class ConvDiscriminator(torch.nn.Module):
    """A discriminator that uses conv layers."""
    def __init__(self, hidden_dims=[], kernel_size=3):
        super(ConvDiscriminator, self).__init__()
        prev_dim = None
        for i, dim in enumerate(hidden_dims):
            if prev_dim is None:
                self.add_module(f"conv{i}", torch.nn.LazyConv2d(dim, kernel_size))
            else:
                self.add_module(f"conv{i}", torch.nn.Conv2d(prev_dim, dim, kernel_size))
            self.add_module(f"relu{i}", torch.nn.ReLU())
            prev_dim = dim
        self.add_module(f"pool", torch.nn.AdaptiveAvgPool2d(1))
        self.add_module(f"flatten", torch.nn.Flatten())
        self.add_module(f"classifier", torch.nn.LazyLinear(1))

    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x
    
class FCDiscriminator(torch.nn.module):
    """A discriminator that uses fully connected layers."""
    def __init__(self, hidden_dims=[]):
        super(FCDiscriminator, self).__init__()
        self.add_module(f"flatten", torch.nn.Flatten())
        prev_dim = None
        for i, dim in enumerate(hidden_dims):
            if prev_dim is None:
                self.add_module(f"fc{i}", torch.nn.LazyLinear(dim))
            else:
                self.add_module(f"fc{i}", torch.nn.Linear(prev_dim, dim))
            self.add_module(f"relu{i}", torch.nn.ReLU())
            prev_dim = dim
        self.add_module(f"classifier", torch.nn.LazyLinear(1))

    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x