from functools import partial 

from detectron2 import model_zoo
from detectron2.config import instantiate
from detectron2.modeling import SwinTransformer
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY


@BACKBONE_REGISTRY.register()
def build_swinb_fpn_backbone(cfg, input_shape):
    """Build a Swin-B FPN backbone using a cfg file. (Detectron2 currently only supports this backbone with LazyConfig.)
    See: detectron2/projects/ViTDet/configs/COCO/cascade_mask_rcnn_swin_b_in21k_50ep.py """
    bottom_up = SwinTransformer(
        depths=[2, 2, 18, 2],
        drop_path_rate=0.4,
        embed_dim=128,
        num_heads=[4, 8, 16, 32],
    )
    in_features = ("p0", "p1", "p2", "p3")
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
        # square_pad=1024 # Default in VitDet comparisons
    )
    return backbone

@BACKBONE_REGISTRY.register()
def build_vitdet_b_backbone(cfg, input_shape):
    backbone = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model.backbone
    backbone.square_pad = 0 # disable square padding
    return instantiate(backbone)

def get_adamw_optim(model, params={}, include_vit_lr_decay=False):
    """See detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py
    and detectron2/projects/ViTDet/configs/COCO/cascade_mask_rcnn_swin_b_in21k_50ep.py"""
    optimizer = model_zoo.get_config("common/optim.py").AdamW
    # From VitDet paper: We also use a layer-wise lr decay [10][2] of 0.7/0.8/0.9 for ViT-B/L/H with 
    # MAE pre-training, which has a small gain of up to 0.3 AP; **we have not seen this gain for 
    # hierarchical backbones or ViT with supervised pre-training.**
    # Thus, disabling the following line by default, since we pretrain with COCO.
    if include_vit_lr_decay:
        optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7)
    optimizer.params.overrides = { "pos_embed": {"weight_decay": 0.0}, # VitDet
                                   "relative_position_bias_table": {"weight_decay": 0.0}, # Swin Transformer
                                }
    for p in params:
        setattr(optimizer, p, params[p])
    optimizer.params.model = model
    return instantiate(optimizer)

def get_swinb_optim(model):
    """See detectron2/projects/ViTDet/configs/COCO/cascade_mask_rcnn_swin_b_in21k_50ep.py"""
    return get_adamw_optim(model, params={"lr": 4e-5, "weight_decay": 0.05})