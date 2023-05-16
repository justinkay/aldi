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

from typing import List, Tuple
import torch

from detectron2.layers import batched_nms, cat
from detectron2.structures import Boxes
from detectron2.modeling.proposal_generator.proposal_utils import _is_tracing
from instances import FreeInstances


def find_top_rpn_proposals(
        proposals: List[torch.Tensor],
        pred_objectness_logits: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
        nms_thresh: float,
        pre_nms_topk: int,
        post_nms_topk: int,
        min_box_size: float,
        training: bool,
        pred_anchor_deltas_sigma=None,
):
    """
    For each feature map, select the `pre_nms_topk` highest scoring proposals,
    apply NMS, clip proposals, and remove small boxes. Return the `post_nms_topk`
    highest scoring proposals among all the feature maps for each image.

    Args:
        proposals (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A, 4).
            All proposal predictions on the feature maps.
        pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A).
        image_sizes (list[tuple]): sizes (h, w) for each image
        nms_thresh (float): IoU threshold.sh to use for NMS
        pre_nms_topk (int): number of top k scoring proposals to keep before applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is per
            feature map.
        post_nms_topk (int): number of top k scoring proposals to keep after applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is total,
            over all feature maps.
        min_box_size (float): minimum proposal box side length in pixels (absolute units
            wrt input images).
        training (bool): True if proposals are to be used in training, otherwise False.
            This arg exists only to support a legacy bug; look for the "NB: Legacy bug ..."
            comment.

    Returns:
        list[Instances]: list of N Instances. The i-th Instances
            stores post_nms_topk object proposals for image i, sorted by their
            objectness score in descending order.
    """
    num_images = len(image_sizes)
    device = proposals[0].device

    # 1. Select top-k anchor for every level and every image
    topk_scores = []  # #lvl Tensor, each of shape N x topk
    topk_proposals = []
    level_ids = []  # #lvl Tensor, each of shape (topk,)
    topk_pred_anchor_deltas_sigma = []
    batch_idx = torch.arange(num_images, device=device)
    if pred_anchor_deltas_sigma is None:
        pred_anchor_deltas_sigma = proposals
    for level_id, (proposals_i, logits_i, pred_anchor_deltas_sigma_i) in enumerate(
            zip(proposals, pred_objectness_logits, pred_anchor_deltas_sigma)):
        Hi_Wi_A = logits_i.shape[1]
        if isinstance(Hi_Wi_A, torch.Tensor):  # it's a tensor in tracing
            num_proposals_i = torch.clamp(Hi_Wi_A, max=pre_nms_topk)
        else:
            num_proposals_i = min(Hi_Wi_A, pre_nms_topk)

        # sort is faster than topk: https://github.com/pytorch/pytorch/issues/22812
        # topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)
        logits_i, idx = logits_i.sort(descending=True, dim=1)
        topk_scores_i = logits_i.narrow(1, 0, num_proposals_i)
        topk_idx = idx.narrow(1, 0, num_proposals_i)

        # each is N x topk
        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # N x topk x 4
        if pred_anchor_deltas_sigma is not None:
            topk_pred_anchor_deltas_sigma_i = pred_anchor_deltas_sigma_i.narrow(1, 0, num_proposals_i)
            topk_pred_anchor_deltas_sigma.append(topk_pred_anchor_deltas_sigma_i)

        topk_proposals.append(topk_proposals_i)
        topk_scores.append(topk_scores_i)
        level_ids.append(torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device))

    # 2. Concat all levels together
    topk_scores = cat(topk_scores, dim=1)
    topk_proposals = cat(topk_proposals, dim=1)
    level_ids = cat(level_ids, dim=0)
    if pred_anchor_deltas_sigma is not None:
        topk_pred_anchor_deltas_sigma = cat(topk_pred_anchor_deltas_sigma, 1)

    # 3. For each image, run a per-level NMS, and choose topk results.
    results: List[FreeInstances] = []
    for n, image_size in enumerate(image_sizes):
        boxes = Boxes(topk_proposals[n])
        scores_per_img = topk_scores[n]
        lvl = level_ids
        if pred_anchor_deltas_sigma is not None:
            pred_anchor_deltas_sigma_per_img = topk_pred_anchor_deltas_sigma[n]

        valid_mask = torch.isfinite(boxes.tensor).all(dim=1) & torch.isfinite(scores_per_img)
        if not valid_mask.all():
            if training:
                raise FloatingPointError(
                    "Predicted boxes or scores contain Inf/NaN. Training has diverged."
                )
            boxes = boxes[valid_mask]
            scores_per_img = scores_per_img[valid_mask]
            lvl = lvl[valid_mask]
            if pred_anchor_deltas_sigma is not None:
                pred_anchor_deltas_sigma_per_img = pred_anchor_deltas_sigma_per_img[valid_mask]
        boxes.clip(image_size)

        # filter empty boxes
        keep = boxes.nonempty(threshold=min_box_size)
        if _is_tracing() or keep.sum().item() != len(boxes):
            boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], lvl[keep]
        if pred_anchor_deltas_sigma is not None:
            pred_anchor_deltas_sigma_per_img = pred_anchor_deltas_sigma_per_img[keep]
            pred_anchor_deltas_sigma_per_img = torch.sigmoid(pred_anchor_deltas_sigma_per_img)
            sigma = 1 - pred_anchor_deltas_sigma_per_img.sum(-1) / 4.
            scores_per_img *= sigma

        keep = batched_nms(boxes.tensor, scores_per_img, lvl, nms_thresh)
        # In Detectron1, there was different behavior during training vs. testing.
        # (https://github.com/facebookresearch/Detectron/issues/459)
        # During training, topk is over the proposals from *all* images in the training batch.
        # During testing, it is over the proposals for each image separately.
        # As a result, the training behavior becomes batch-dependent,
        # and the configuration "POST_NMS_TOPK_TRAIN" end up relying on the batch size.
        # This bug is addressed in Detectron2 to make the behavior independent of batch size.
        keep = keep[:post_nms_topk]  # keep is already sorted

        res = FreeInstances(image_size)
        res.proposal_boxes = boxes[keep]
        res.objectness_logits = scores_per_img[keep]
        results.append(res)
    return results