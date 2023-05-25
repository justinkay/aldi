from detectron2.modeling import SwinTransformer
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
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