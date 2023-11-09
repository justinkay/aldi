# code adapted from https://github.com/Shengcao-Cao/CMT
import warnings
import torch
import torch.nn as nn
from torchvision.ops import roi_align


def data2boxes(data):
    boxes = []
    for i in range(len(data)):
        boxes_i = data[i]['instances'].gt_boxes.tensor
        if boxes_i.shape[0]:
            indices = i * torch.ones((boxes_i.shape[0], 1), dtype=boxes_i.dtype, device=boxes_i.device)
            boxes_i = torch.cat([indices, boxes_i], dim=1)
            boxes.append(boxes_i)
    if len(boxes):
        boxes = torch.cat(boxes, dim=0)
        return boxes
    else:
        return None


def data2labels(data):
    labels = []
    for i in range(len(data)):
        labels_i = data[i]['instances'].gt_classes
        if labels_i.shape[0]:
            labels.append(labels_i)
    labels = torch.cat(labels, dim=0)
    return labels


def locate_feature_roialign(feature_map, boxes, image_width, image_height):
    selected_features = []
    sx = feature_map.shape[3] / image_width
    sy = feature_map.shape[2] / image_height
    if len(boxes):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boxes_level = torch.tensor(boxes, device=feature_map.device)
        boxes_level[:, 1] *= sx
        boxes_level[:, 2] *= sy
        boxes_level[:, 3] *= sx
        boxes_level[:, 4] *= sy
        selected_features_level = roi_align(feature_map, boxes_level, output_size=1, aligned=True)
        selected_features_level = torch.flatten(selected_features_level, start_dim=1)
        selected_features = selected_features_level
    else:
        selected_features = None
    return selected_features


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, weights=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        if weights is not None:
            loss = (loss.view(anchor_count, batch_size) * weights).sum() / weights.sum()
        else:
            loss = loss.view(anchor_count, batch_size).mean()

        return loss


def get_cmt_losses(unlabeled_strong, unlabeled_weak, pseudolabeled_data, features_student, features_teacher, cmt_loss_weight):
    unlabel_data_q, unlabel_data_k, pslabel_data_q = unlabeled_strong, unlabeled_weak, pseudolabeled_data,
    all_unlabel_data = pslabel_data_q

    feature_levels = ['p4', 'p5', 'p6']  # TODO: make configurable

    supconloss = SupConLoss(contrast_mode='one')

    # 7. CMT: object-level contrastive learning
    record_dict = dict()
    boxes = data2boxes(all_unlabel_data)
    image_width = all_unlabel_data[0]['image'].shape[2]
    image_height = all_unlabel_data[0]['image'].shape[1]

    # filter objects that are too different in two views
    if boxes is not None:
        flags = []
        for i in range(boxes.shape[0]):
            box_i = boxes[i].to(torch.int)
            image_index = box_i[0]
            x1 = box_i[1]
            y1 = box_i[2]
            x2 = box_i[3]
            y2 = box_i[4]
            image_q_patch = unlabel_data_q[image_index]['image'][:, y1:y2, x1:x2].to(torch.float)
            image_k_patch = unlabel_data_k[image_index]['image'][:, y1:y2, x1:x2].to(torch.float)
            diff = (image_q_patch - image_k_patch).absolute().flatten()
            ratio = (diff > 40).sum() / diff.numel()
            if ratio > 0.5:
                flags.append(0)
            else:
                flags.append(1)
    else:
        flags = [0]

    if sum(flags):
        # build contrastive loss
        for feature_level in feature_levels:
            # get student and teacher features for objects
            object_features_student = locate_feature_roialign(features_student[feature_level], boxes, image_width, image_height)
            object_features_teacher = locate_feature_roialign(features_teacher[feature_level], boxes, image_width, image_height)
            object_features_student = torch.nn.functional.normalize(object_features_student, dim=1)
            object_features_teacher = torch.nn.functional.normalize(object_features_teacher, dim=1)

            # compute contrastive loss
            object_features_all = torch.stack([object_features_student, object_features_teacher], dim=1)
            object_labels = data2labels(all_unlabel_data)

            # exclude unused objects
            flags = [bool(x) for x in flags]
            object_features_all = object_features_all[flags]
            object_labels = object_labels[flags]
            loss_contrastive_object = supconloss(object_features_all, object_labels)

            # record contrastive loss
            record_dict['loss_contrastive_object' + '_' + feature_level] = loss_contrastive_object * cmt_loss_weight

    else:
        for feature_level in feature_levels:
            record_dict['loss_contrastive_object' + '_' + feature_level] = torch.tensor(0.0)

    return record_dict