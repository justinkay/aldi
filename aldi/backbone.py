from functools import partial
import math
import warnings
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from detectron2 import model_zoo
from detectron2.config import instantiate
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.utils import get_abs_pos
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
from detectron2.layers import ShapeSpec


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

@BACKBONE_REGISTRY.register()
def build_vitdet_l_backbone(cfg, input_shape):
    backbone = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model.backbone
    backbone.square_pad = 0 # disable square padding

    # ViT-L stuff
    backbone.net.embed_dim = 1024
    backbone.net.depth = 24
    backbone.net.num_heads = 16
    backbone.net.drop_path_rate = 0.4
    # 5, 11, 17, 23 for global attention
    backbone.net.window_block_indexes = (
        list(range(0, 5)) + list(range(6, 11)) + list(range(12, 17)) + list(range(18, 23))
    )

    backbone = instantiate(backbone)

    # TODO doesn't work yet
    # backbone.net.forward = partial(checkpointed_vit_forward, backbone.net, cfg.VIT.USE_ACT_CHECKPOINT)
   
    return backbone

def get_adamw_optim(model, params={}, include_vit_lr_decay=False, vit_size='b'):
    """See detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py"""
    optimizer = model_zoo.get_config("common/optim.py").AdamW
    # From VitDet paper: We also use a layer-wise lr decay [10][2] of 0.7/0.8/0.9 for ViT-B/L/H with 
    # MAE pre-training, which has a small gain of up to 0.3 AP; **we have not seen this gain for 
    # hierarchical backbones or ViT with supervised pre-training.**
    # Thus, disabling the following line by default, since we pretrain with COCO.
    if include_vit_lr_decay:
        if vit_size == 'b':
            optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7)
        elif vit_size == 'l':
            optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, lr_decay_rate=0.8, num_layers=24)
        else:
            raise ValueError(f"ViT size {vit_size} not supported.")
    optimizer.params.overrides = { "pos_embed": {"weight_decay": 0.0}, }
    for p in params:
        setattr(optimizer, p, params[p])
    optimizer.params.model = model
    return instantiate(optimizer)


###################################################################################################################
# ConvNeXt implementation 
# taken from: https://github.com/facebookresearch/ConvNeXt/pull/43/commits/c1ffd00355b6213f55b73263a99e03b3a126495c
###################################################################################################################

# From timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/weight_init.py#L43
def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor

# From timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/weight_init.py#L43
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)

# from timm: https://github.com/huggingface/pytorch-image-models/blob/131518c15cef20aa6cfe3c6831af3a1d0637e3d1/timm/layers/drop.py#L170
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

# from timm: https://github.com/huggingface/pytorch-image-models/blob/131518c15cef20aa6cfe3c6831af3a1d0637e3d1/timm/layers/drop.py#L170
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class ConvNextBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(Backbone):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
        out_features (tuple(int)): Stage numbers of the outputs given to the Neck.
    """
    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_features=None):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )

        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)


        self.num_layers = len(depths)
        num_features = [int(dims[i] * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        self._out_features = out_features

        self._out_feature_strides = {}
        self._out_feature_channels = {}

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        strides = [4,4,4,4] 
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNextBlock(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

            self._out_feature_channels[i] = dims[i]
            self._out_feature_strides[i] = strides[i] * 2 ** i

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def forward_features(self, x):
        outs ={} 
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self._out_features:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                out = x_out.contiguous()
                stage_name = i
                outs[stage_name] = out

        return outs  # {"stage%d" % (i+2,): out for i, out in enumerate(outs)} #tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

@BACKBONE_REGISTRY.register()
def build_convnext_backbone(cfg, input_shape):
    """
    Create a ConvNeXt instance from config.
    Returns:
        VoVNet: a :class:`VoVNet` instance.
    """
    return ConvNeXt(
        in_chans=input_shape.channels,
        depths=cfg.MODEL.CONVNEXT.DEPTHS,
        dims=cfg.MODEL.CONVNEXT.DIMS,
        drop_path_rate=cfg.MODEL.CONVNEXT.DROP_PATH_RATE,
        layer_scale_init_value=cfg.MODEL.CONVNEXT.LAYER_SCALE_INIT_VALUE,
        out_features=cfg.MODEL.CONVNEXT.OUT_FEATURES
    )

@BACKBONE_REGISTRY.register()
def build_convnext_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_convnext_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone