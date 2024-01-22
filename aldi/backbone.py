from functools import partial 
from torch.utils.checkpoint import checkpoint

from detectron2 import model_zoo
from detectron2.config import instantiate
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.utils import get_abs_pos


# TODO: hacky: patch forward method of the ViT backbone to enable
# vanilla PyTorch non-reantrant checkpointing which works with DDP
def checkpointed_vit_forward(self, use_checkpointing, x):
    x = self.patch_embed(x)
    if self.pos_embed is not None:
        x = x + get_abs_pos(
            self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
        )

    for blk in self.blocks:
        if use_checkpointing and self.training:
            x = checkpoint(blk, x, use_reentrant=False)
        else:
            x = blk(x)

    outputs = {self._out_features[0]: x.permute(0, 3, 1, 2)}
    return outputs

@BACKBONE_REGISTRY.register()
def build_vitdet_b_backbone(cfg, input_shape):
    backbone = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model.backbone
    backbone.square_pad = 0 # disable square padding
    backbone = instantiate(backbone)
    backbone.net.forward = partial(checkpointed_vit_forward, backbone.net, cfg.VIT.USE_ACT_CHECKPOINT)
    return backbone

def get_adamw_optim(model, params={}, include_vit_lr_decay=False):
    """See detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py"""
    optimizer = model_zoo.get_config("common/optim.py").AdamW
    # From VitDet paper: We also use a layer-wise lr decay [10][2] of 0.7/0.8/0.9 for ViT-B/L/H with 
    # MAE pre-training, which has a small gain of up to 0.3 AP; **we have not seen this gain for 
    # hierarchical backbones or ViT with supervised pre-training.**
    # Thus, disabling the following line by default, since we pretrain with COCO.
    if include_vit_lr_decay:
        optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7)
    optimizer.params.overrides = { "pos_embed": {"weight_decay": 0.0}, }
    for p in params:
        setattr(optimizer, p, params[p])
    optimizer.params.model = model
    return instantiate(optimizer)