from __future__ import print_function
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.modeling.poolers import ROIPooler, assign_boxes_to_levels

class SADA(torch.nn.Module):
    def __init__(self, cfg):
        super(SADA, self).__init__()

        self.cfg = cfg.clone()

        # stage_index = 4
        # stage2_relative_factor = 2 ** (stage_index - 1)
        # res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_ins_inputs = cfg.MODEL.ROI_BOX_HEAD.FC_DIM #.MLP_HEAD_DIM #if cfg.MODEL.RPN.USE_FPN else res2_out_channels * stage2_relative_factor

        self.USE_FPN = True # cfg.MODEL.RPN.USE_FPN
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)

        self.consit_weight = self.cfg.DOMAIN_ADAPT.ALIGN.SADA_COS_WEIGHT

        self.grl_img = GradientScalarLayer(-1.0 * self.cfg.DOMAIN_ADAPT.ALIGN.SADA_IMG_GRL_WEIGHT)
        self.grl_ins = GradientScalarLayer(-1.0 * self.cfg.DOMAIN_ADAPT.ALIGN.SADA_INS_GRL_WEIGHT)
        self.grl_img_consist = GradientScalarLayer(self.consit_weight * self.cfg.DOMAIN_ADAPT.ALIGN.SADA_IMG_GRL_WEIGHT)
        self.grl_ins_consist = GradientScalarLayer(self.consit_weight * self.cfg.DOMAIN_ADAPT.ALIGN.SADA_INS_GRL_WEIGHT)

        # in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        # pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        # pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        # sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        # pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        in_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS # ? .MODEL.RESNETS.STEM_OUT_CHANNELS # ? cfg.MODEL.BACKBONE.OUT_CHANNELS

        self.imghead = SADAImgHead(in_channels)
        self.loss_evaluator = make_sada_heads_loss_evaluator(cfg)

        # scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        self.lvl_min = self.loss_evaluator.pooler.min_level #-torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        self.lvl_max = self.loss_evaluator.pooler.max_level #-torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        # self.map_levels = LevelMapper(lvl_min, lvl_max) #canonical_scale=224, canonical_level=4, eps=1e-7
        self.inshead = SADAInsHead(num_ins_inputs)

    def forward(self, img_features, da_ins_feature, da_ins_labels, da_proposals, img_targets):
        """
        Arguments:
            img_features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            da_ins_feature (Tensor): instance feature vectors extracted according to da_proposals
            da_ins_labels (Tensor): domain labels for instance feature vectors
            da_proposals (list[BoxList]): randomly selected proposal boxes
            targets (list[BoxList): ground-truth boxes present in the image (optional)
        Returns:
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        if not self.USE_FPN:
            da_ins_feature = self.avgpool(da_ins_feature)
        da_ins_feature = da_ins_feature.view(da_ins_feature.size(0), -1)

        img_grl_fea = [self.grl_img(fea) for fea in img_features]
        ins_grl_fea = self.grl_ins(da_ins_feature)
        img_grl_consist_fea = [self.grl_img_consist(fea) for fea in img_features]
        ins_grl_consist_fea = self.grl_ins_consist(da_ins_feature)

        # instance alignment
        # levels = self.map_levels(da_proposals)
        levels = assign_boxes_to_levels(da_proposals, self.lvl_min, self.lvl_max, 
                                        canonical_box_size=224, canonical_level=4)
        da_ins_features = self.inshead(ins_grl_fea, levels)
        da_ins_consist_features = self.inshead(ins_grl_consist_fea, levels)

        da_ins_consist_features = da_ins_consist_features.sigmoid()

        # image alignment
        da_img_features = self.imghead(img_grl_fea)
        da_img_consist_features = self.imghead(img_grl_consist_fea)
        da_img_consist_features = [fea.sigmoid() for fea in da_img_consist_features]

        if self.training:
            da_img_loss, da_ins_loss, da_consistency_loss = self.loss_evaluator(
                da_proposals, da_img_features, da_ins_features, da_img_consist_features, da_ins_consist_features,
                da_ins_labels, img_targets)

            losses = {
                "loss_da_image": da_img_loss,
                "loss_da_instance": da_ins_loss,
                "loss_da_consistency": da_consistency_loss}

            return losses

        return {}

class SADALossComputation(object):
    """
    This class computes the DA loss.
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()

        # in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        # pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        # pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        # sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        # pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # box_in_features=["p2", "p3", "p4", "p5"],
        # box_pooler=L(ROIPooler)(
        #     output_size=7,
        #     scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
        #     sampling_ratio=0,
        #     pooler_type="ROIAlignV2",
        # ),

        # TODO: how to get this programatically?
        # input_shape = {'res2': ShapeSpec(channels=256, height=None, width=None, stride=4), 
        #                'res3': ShapeSpec(channels=512, height=None, width=None, stride=8), 
        #                'res4': ShapeSpec(channels=1024, height=None, width=None, stride=16), 
        #                'res5': ShapeSpec(channels=2048, height=None, width=None, stride=32)}
        input_shape = {'p2': ShapeSpec(channels=256, height=None, width=None, stride=4), 
                       'p3': ShapeSpec(channels=512, height=None, width=None, stride=8), 
                       'p4': ShapeSpec(channels=1024, height=None, width=None, stride=16), 
                       'p5': ShapeSpec(channels=2048, height=None, width=None, stride=32),
                       'p6': ShapeSpec(channels=2048, height=None, width=None, stride=64),}
        in_features = ["p2", "p3", "p4", "p5", "p6"] # cfg.MODEL.ROI_HEADS.IN_FEATURES
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = tuple(1.0 / input_shape[k].stride for k in in_features) # cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = ROIPooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
            pooler_type="ROIAlignV2"
        )

        self.pooler = pooler
        self.avgpool = nn.AvgPool2d(kernel_size=resolution, stride=resolution)

    # def __call__(self, proposals, da_img, da_ins, da_img_consist, da_ins_consist, da_ins_labels, targets, da_img_features_joint):
    def __call__(self, proposals, da_img, da_ins, da_img_consist, da_ins_consist, da_ins_labels, img_targets):
        """
        Arguments:
            proposals (list[BoxList])
            da_img (list[Tensor])
            da_img_consist (list[Tensor])
            da_ins (Tensor)
            da_ins_consist (Tensor)
            da_ins_labels (Tensor)
            targets (list[BoxList])
        Returns:
            da_img_loss (Tensor)
            da_ins_loss (Tensor)
            da_consist_loss (Tensor)
        """

        # masks = self.prepare_masks(targets)
        # masks = torch.cat(masks, dim=0)
        masks = img_targets # replaced this parameter in order to pass in domain labels directly

        # for each feature level, permute the outputs to make them be in the
        # same format as the labels. Note that the labels are computed for
        # all feature levels concatenated, so we keep the same representation
        # for the image-level domain alignment

        # da_img_loss = []
        # for da_img_per_level in da_img:
        #     N, A, H, W = da_img_per_level.shape
        #     da_img_per_level = da_img_per_level.permute(0, 2, 3, 1)
        #
        #     da_img_label_per_level = torch.zeros_like(da_img_per_level, dtype=torch.float32)
        #     da_img_label_per_level[masks, :] = 1
        #
        #     da_img_per_level = da_img_per_level.reshape(N, -1)
        #     da_img_label_per_level = da_img_label_per_level.reshape(N, -1)
        #
        #     da_img_loss.append(F.binary_cross_entropy_with_logits(da_img_per_level, da_img_label_per_level)/len(da_img))
        #
        # da_img_loss = torch.sum(torch.stack(da_img_loss))

        # new da img
        _, _, H, W = da_img[0].shape
        up_sample = nn.Upsample(size=(H, W), mode='bilinear', align_corners=True)
        upsampled_loss = []
        for i, feat in enumerate(da_img):
            feat = da_img[i]
            feat = up_sample(feat)
            da_img_label_per_level = torch.zeros_like(feat, dtype=torch.float32)
            da_img_label_per_level[masks, :] = 1
            lv_loss = F.binary_cross_entropy_with_logits\
                (feat, da_img_label_per_level, reduction='none')
            upsampled_loss.append(lv_loss)

        da_img_loss = torch.stack(upsampled_loss)
        # da_img_loss, _ = torch.median(da_img_loss, dim=0)
        # da_img_loss, _ = torch.max(da_img_loss, dim=0)
        # da_img_loss, _ = torch.min(da_img_loss, dim=0)
        da_img_loss = da_img_loss.mean()

        # da img joint
        # feat = da_img_features_joint[0]
        # feat = up_sample(feat)
        # da_img_label_per_level = torch.zeros_like(feat, dtype=torch.float32)
        # da_img_label_per_level[masks, :] = 1
        # joint_loss = F.binary_cross_entropy_with_logits \
        #     (feat, da_img_label_per_level)

        #ins da
        da_ins_loss = F.binary_cross_entropy_with_logits(
            torch.squeeze(da_ins), da_ins_labels.type(torch.cuda.FloatTensor)
        )

        da_img_rois_probs = self.pooler(da_img_consist, proposals)
        da_img_rois_probs_pool = self.avgpool(da_img_rois_probs)
        da_img_rois_probs_pool = da_img_rois_probs_pool.view(da_img_rois_probs_pool.size(0), -1)

        # da_consist_loss = consistency_loss(da_img_consist, da_ins_consist, da_ins_labels, size_average=True)
        da_consist_loss = F.l1_loss(da_img_rois_probs_pool, da_ins_consist)

        return da_img_loss, da_ins_loss, da_consist_loss

def make_sada_heads_loss_evaluator(cfg):
    loss_evaluator = SADALossComputation(cfg)
    return loss_evaluator

class _GradientScalarLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.weight = weight
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return ctx.weight*grad_input, None

gradient_scalar = _GradientScalarLayer.apply

class GradientScalarLayer(torch.nn.Module):
    def __init__(self, weight):
        super(GradientScalarLayer, self).__init__()
        self.weight = weight

    def forward(self, input):
        return gradient_scalar(input, self.weight)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "weight=" + str(self.weight)
        tmpstr += ")"
        return tmpstr

class SADAImgHead(nn.Module):
    """
    Adds a simple Image-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            USE_FPN (boolean): whether FPN feature extractor is used
        """
        super(SADAImgHead, self).__init__()

        self.da_img_conv1_layers = []
        self.da_img_conv2_layers = []
        for idx in range(5):
            conv1_block = "da_img_conv1_level{}".format(idx)
            conv2_block = "da_img_conv2_level{}".format(idx)
            conv1_block_module = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1)
            conv2_block_module = nn.Conv2d(512, 1, kernel_size=1, stride=1)
            for module in [conv1_block_module, conv2_block_module]:
                # Caffe2 implementation uses XavierFill, which in fact
                # corresponds to kaiming_uniform_ in PyTorch
                torch.nn.init.normal_(module.weight, std=0.001)
                torch.nn.init.constant_(module.bias, 0)
            self.add_module(conv1_block, conv1_block_module)
            self.add_module(conv2_block, conv2_block_module)
            self.da_img_conv1_layers.append(conv1_block)
            self.da_img_conv2_layers.append(conv2_block)


    def forward(self, x):
        img_features = []

        for feature, conv1_block, conv2_block in zip(
                x, self.da_img_conv1_layers, self.da_img_conv2_layers
        ):
            inner_lateral = getattr(self, conv1_block)(feature)
            last_inner = F.relu(inner_lateral)
            img_features.append(getattr(self, conv2_block)(last_inner))
        return img_features

class SADAInsHead(nn.Module):
    """
    Adds a simple Instance-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(SADAInsHead, self).__init__()

        self.da_ins_fc1_layers = []
        self.da_ins_fc2_layers = []
        self.da_ins_fc3_layers = []

        for idx in range(4):
            fc1_block = "da_ins_fc1_level{}".format(idx)
            fc2_block = "da_ins_fc2_level{}".format(idx)
            fc3_block = "da_ins_fc3_level{}".format(idx)
            fc1_block_module = nn.Linear(in_channels, 1024)
            fc2_block_module = nn.Linear(1024, 1024)
            fc3_block_module = nn.Linear(1024, 1)
            for module in [fc1_block_module, fc2_block_module, fc3_block_module]:
                # Caffe2 implementation uses XavierFill, which in fact
                # corresponds to kaiming_uniform_ in PyTorch
                nn.init.normal_(module.weight, std=0.01)
                nn.init.constant_(module.bias, 0)
            self.add_module(fc1_block, fc1_block_module)
            self.add_module(fc2_block, fc2_block_module)
            self.add_module(fc3_block, fc3_block_module)
            self.da_ins_fc1_layers.append(fc1_block)
            self.da_ins_fc2_layers.append(fc2_block)
            self.da_ins_fc3_layers.append(fc3_block)

    def forward(self, x, levels=None):

        dtype, device = x.dtype, x.device

        result = torch.zeros((x.shape[0], 1),
            dtype=dtype, device=device,)

        for level, (fc1_da, fc2_da, fc3_da) in \
                enumerate(zip(self.da_ins_fc1_layers,
                              self.da_ins_fc2_layers, self.da_ins_fc3_layers)):

            idx_in_level = torch.nonzero(levels == level).squeeze(1)

            if len(idx_in_level) > 0:
                xs = x[idx_in_level, :]

                xs = F.relu(getattr(self, fc1_da)(xs))
                xs = F.dropout(xs, p=0.5, training=self.training)

                xs = F.relu(getattr(self, fc2_da)(xs))
                xs = F.dropout(xs, p=0.5, training=self.training)

                result[idx_in_level] = getattr(self, fc3_da)(xs)

            # JK: this return is here in sa-da-faster and MIC, but it seems like a bug?
            return result

        # return result # JK: Added this one, which seems correct; but removing to try and reproduce original results

def grad_reverse(x):
    return _GradientScalarLayer.apply(x, -1.0)