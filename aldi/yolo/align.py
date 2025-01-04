import torch
import torch.nn.functional as F

from detectron2.config import configurable

from aldi.align import ConvDiscriminator, ALIGN_MIXIN_REGISTRY
from aldi.helpers import SaveIO, grad_reverse

from .libs.Yolo_Detectron2.yolo_detectron2 import Yolo


@ALIGN_MIXIN_REGISTRY.register()
class YoloAlignMixin(Yolo):
    @configurable
    def __init__(
        self,
        *,
        img_da_enabled: bool = False,
        img_da_layer: str = None,
        img_da_weight: float = 0.0,
        ins_da_enabled: bool = False,
        ins_da_weight: float = 0.0,
        **kwargs
    ):
        super(YoloAlignMixin, self).__init__(**kwargs)

        if ins_da_enabled:
            raise NotImplementedError()

        self.img_da_layer = img_da_layer
        self.img_da_weight = img_da_weight

        self.img_align = ConvDiscriminator(768, hidden_dims=[256]) if img_da_enabled else None # TODO dims; same as Yolo

        # register hooks so we can grab output of sub-modules
        # allow for img_da_layer to specify either p3, p4, or p5 as the alignment layer
        self.backbone_io = {
            "p3": SaveIO(),
            "p4": SaveIO(),
            "p5": SaveIO(),
        }
        self.model[17].register_forward_hook(self.backbone_io["p3"])
        self.model[20].register_forward_hook(self.backbone_io["p4"])
        self.model[23].register_forward_hook(self.backbone_io["p5"])

    @classmethod
    def from_config(cls, cfg):
        ret = super(YoloAlignMixin, cls).from_config(cfg)

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
            if do_align and self.img_align:
                if self.img_align:
                    # extract needed info for alignment: domain labels, image features, instance features
                    domain_label = 1 if labeled else 0
                    features = self.backbone_io[self.img_da_layer].output
                    features = grad_reverse(features)
                    domain_preds = self.img_align(features)
                    loss = F.binary_cross_entropy_with_logits(domain_preds, torch.FloatTensor(domain_preds.data.size()).fill_(domain_label).to(features.device))
                    output["loss_da_img"] = self.img_da_weight * loss
            elif self.img_align:
                # need to utilize the modules at some point during the forward pass or PyTorch complains.
                # this is only an issue when cfg.SOLVER.BACKWARD_AT_END=False, because intermediate backward()
                # calls may not have used alignment heads
                # see: https://github.com/pytorch/pytorch/issues/43259#issuecomment-964284292
                fake_output = 0
                for aligner in [self.img_align]:
                    if aligner is not None:
                        fake_output += sum([p.sum() for p in aligner.parameters()]) * 0
                output["_da"] = fake_output
        return output