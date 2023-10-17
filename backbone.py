from functools import partial 

import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from detectron2 import model_zoo
from detectron2.config import instantiate
from detectron2.modeling import SwinTransformer
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.utils import get_abs_pos


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
    backbone = instantiate(backbone)

    # TODO: hacky: patch forward method of the ViT backbone to enable
    # vanilla PyTorch non-reantrant checkpointing which works with DDP
    def forward(self, x):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )

        for blk in self.blocks:
            if cfg.VIT.USE_ACT_CHECKPOINT:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        outputs = {self._out_features[0]: x.permute(0, 3, 1, 2)}
        return outputs

    backbone.net.forward = lambda x: forward(backbone.net, x)
    return backbone

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

###
### The rest of this is taken from Adaptive Teacher VGG implementation ###
###
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.nn as nn
from typing import Union, List, Dict, Any, cast
from detectron2.modeling.backbone import (
    Backbone,
    build_resnet_backbone,
    BACKBONE_REGISTRY
)
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool

def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs: Dict[str, List[Union[str, int]]] = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class vgg_backbone(Backbone):
    """
    Backbone (bottom-up) for FBNet.

    Hierarchy:
        trunk0:
            xif0_0
            xif0_1
            ...
        trunk1:
            xif1_0
            xif1_1
            ...
        ...

    Output features:
        The outputs from each "stage", i.e. trunkX.
    """

    def __init__(self, cfg):
        super().__init__()

        self.vgg = make_layers(cfgs['vgg16'],batch_norm=True)

        self._initialize_weights()
        # self.stage_names_index = {'vgg1':3, 'vgg2':8 , 'vgg3':15, 'vgg4':22, 'vgg5':29}
        _out_feature_channels = [64, 128, 256, 512, 512]
        _out_feature_strides = [2, 4, 8, 16, 32]
        # stages, shape_specs = build_fbnet(
        #     cfg,
        #     name="trunk",
        #     in_channels=cfg.MODEL.FBNET_V2.STEM_IN_CHANNELS
        # )

        # nn.Sequential(*list(self.vgg.features._modules.values())[:14])

        self.stages = [nn.Sequential(*list(self.vgg._modules.values())[0:7]),\
                    nn.Sequential(*list(self.vgg._modules.values())[7:14]),\
                    nn.Sequential(*list(self.vgg._modules.values())[14:24]),\
                    nn.Sequential(*list(self.vgg._modules.values())[24:34]),\
                    nn.Sequential(*list(self.vgg._modules.values())[34:]),]
        self._out_feature_channels = {}
        self._out_feature_strides = {}
        self._stage_names = []

        for i, stage in enumerate(self.stages):
            name = "vgg{}".format(i)
            self.add_module(name, stage)
            self._stage_names.append(name)
            self._out_feature_channels[name] = _out_feature_channels[i]
            self._out_feature_strides[name] = _out_feature_strides[i]

        self._out_features = self._stage_names

        del self.vgg

    def forward(self, x):
        features = {}
        for name, stage in zip(self._stage_names, self.stages):
            x = stage(x)
            # if name in self._out_features:
            #     outputs[name] = x
            features[name] = x
        # import pdb
        # pdb.set_trace()

        return features

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


@BACKBONE_REGISTRY.register() #already register in baseline model
def build_vgg_backbone(cfg, _):
    return vgg_backbone(cfg)


@BACKBONE_REGISTRY.register() #already register in baseline model
def build_vgg_fpn_backbone(cfg, _):
    bottom_up = vgg_backbone(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
    )
    return backbone