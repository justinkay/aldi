# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import math
from typing import List, Tuple
import torch
from fvcore.nn import giou_loss, smooth_l1_loss

from detectron2.layers import cat
from detectron2.structures import Boxes
from detectron2.modeling.box_regression import _dense_box_regression_loss as _d2_dense_box_regression_loss

# Value for clamping large dw and dh predictions. The heuristic is that we clamp
# such that dw and dh are no larger than what would transform a 16px box into a
# 1000px box (based on a small anchor, 16px, and a typical image size, 1000px).
_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)

__all__ = ["Box2BoxTransform", ]


def gaussian_dist_pdf(val, mean, var, eps=1e-7): # 9): # careful with FP16
    simga_constant = 0.3
    return torch.exp(-(val - mean) ** 2.0 / (var + eps) / 2.0) / torch.sqrt(2.0 * np.pi * (var + simga_constant))


def laplace_dist_pdf(val, mean, var, eps=1e-7): # 9): # careful with FP16
    simga_constant = 0.3
    return torch.exp(-torch.abs(val - mean) / torch.sqrt(var + eps)) / torch.sqrt(4.0 * (var + simga_constant))


@torch.jit.script
class Box2BoxTransform(object):
    """
    The box-to-box transform defined in R-CNN. The transformation is parameterized
    by 4 deltas: (dx, dy, dw, dh). The transformation scales the box's width and height
    by exp(dw), exp(dh) and shifts a box's center by the offset (dx * width, dy * height).
    """

    def __init__(
            self, weights: Tuple[float, float, float, float], scale_clamp: float = _DEFAULT_SCALE_CLAMP
    ):
        """
        Args:
            weights (4-element tuple): Scaling factors that are applied to the
                (dx, dy, dw, dh) deltas. In Fast R-CNN, these were originally set
                such that the deltas have unit variance; now they are treated as
                hyperparameters of the system.
            scale_clamp (float): When predicting deltas, the predicted box scaling
                factors (dw and dh) are clamped such that they are <= scale_clamp.
        """
        self.weights = weights
        self.scale_clamp = scale_clamp

    def get_deltas(self, src_boxes, target_boxes):
        """
        Get box regression transformation deltas (dx, dy, dw, dh) that can be used
        to transform the `src_boxes` into the `target_boxes`. That is, the relation
        ``target_boxes == self.apply_deltas(deltas, src_boxes)`` is true (unless
        any delta is too large and is clamped).

        Args:
            src_boxes (Tensor): source boxes, e.g., object proposals
            target_boxes (Tensor): target of the transformation, e.g., ground-truth
                boxes.
        """
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)
        eps = 1e-7

        src_widths = src_boxes[:, 2] - src_boxes[:, 0] #+ eps
        src_heights = src_boxes[:, 3] - src_boxes[:, 1] #+ eps
        src_ctr_x = src_boxes[:, 0] + 0.5 * src_widths
        src_ctr_y = src_boxes[:, 1] + 0.5 * src_heights

        target_widths = target_boxes[:, 2] - target_boxes[:, 0]
        target_heights = target_boxes[:, 3] - target_boxes[:, 1]
        target_ctr_x = target_boxes[:, 0] + 0.5 * target_widths
        target_ctr_y = target_boxes[:, 1] + 0.5 * target_heights

        wx, wy, ww, wh = self.weights
        dx = wx * (target_ctr_x - src_ctr_x) / src_widths
        dy = wy * (target_ctr_y - src_ctr_y) / src_heights
        dw = ww * torch.log(target_widths / src_widths + eps)
        dh = wh * torch.log(target_heights / src_heights + eps)

        deltas = torch.stack((dx, dy, dw, dh), dim=1)
        assert (src_widths > 0).all().item(), "Input boxes to Box2BoxTransform are not valid! (src_widths <= 0)"
        assert (src_heights > 0).all().item(), "Input boxes to Box2BoxTransform are not valid! (src_heights <= 0)"
        return deltas

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        deltas = deltas.float()  # ensure fp32 for decoding precision
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h
        pred_boxes = torch.stack((x1, y1, x2, y2), dim=-1)
        return pred_boxes.reshape(deltas.shape)


def _dense_box_regression_loss(
        anchors: List[Boxes],
        box2box_transform: Box2BoxTransform,
        pred_anchor_deltas: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
        fg_mask: torch.Tensor,
        box_reg_loss_type="smooth_l1",
        smooth_l1_beta=0.0,
        model_type="",
):
    """
    Compute loss for dense multi-level box regression.
    Loss is accumulated over ``fg_mask``.

    Args:
        anchors: #lvl anchor boxes, each is (HixWixA, 4)
        pred_anchor_deltas: #lvl predictions, each is (N, HixWixA, 4)
        gt_boxes: N ground truth boxes, each has shape (R, 4) (R = sum(Hi * Wi * A))
        fg_mask: the foreground boolean mask of shape (N, R) to compute loss on
        box_reg_loss_type (str): Loss type to use. Supported losses: "smooth_l1", "giou".
        smooth_l1_beta (float): beta parameter for the smooth L1 regression loss. Default to
            use L1 loss. Only used when `box_reg_loss_type` is "smooth_l1"
    """
    anchors = type(anchors[0]).cat(anchors).tensor  # (R, 4)
    if box_reg_loss_type == "smooth_l1":
        gt_anchor_deltas = [box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, R, 4)

        if model_type == "GAUSSIAN":
            sigma_xywh = torch.sigmoid(cat(pred_anchor_deltas, dim=1)[..., -4:])[fg_mask]
            mean_xywh = cat(pred_anchor_deltas, dim=1)[..., :4][fg_mask]
            gaussian = gaussian_dist_pdf(mean_xywh,
                                         gt_anchor_deltas[fg_mask], sigma_xywh)
            loss_box_reg_gaussian = - torch.log(gaussian + 1e-7).sum()
            loss_box_reg = loss_box_reg_gaussian
        elif model_type == "LAPLACE":
            sigma_xywh = torch.sigmoid(cat(pred_anchor_deltas, dim=1)[..., -4:])[fg_mask]
            mean_xywh = cat(pred_anchor_deltas, dim=1)[..., :4][fg_mask]
            laplace = laplace_dist_pdf(mean_xywh,
                                       gt_anchor_deltas[fg_mask], sigma_xywh)
            loss_box_reg_laplace = - torch.log(laplace + 1e-7).sum()
            loss_box_reg = loss_box_reg_laplace
        else:
            loss_box_reg = smooth_l1_loss(
                cat(pred_anchor_deltas, dim=1)[fg_mask],
                gt_anchor_deltas[fg_mask],
                beta=smooth_l1_beta,
                reduction="sum",
            )
        return loss_box_reg
    else:
        return _d2_dense_box_regression_loss(anchors, box2box_transform, pred_anchor_deltas, gt_boxes, 
                                             fg_mask, box_reg_loss_type, smooth_l1_beta)