#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Copied directly from detectron2/tools/train_net.py except where noted.
"""
import os
from datetime import timedelta

import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import inference_on_dataset

from aldi.checkpoint import DetectionCheckpointerWithEMA
from aldi.config import add_aldi_config
from aldi.ema import EMA
from aldi.trainer import ALDITrainer
import aldi.datasets # register datasets with Detectron2
import aldi.model # register ALDI R-CNN model with Detectron2
import aldi.backbone # register ViT FPN backbone with Detectron2

try:
    from sklearn.decomposition import PCA
except ModuleNotFoundError:
    print("""
          Feature visualization requires scikit-learn to be installed.
          Please insteall scikit-learn (e.g. run `pip install scikit-learn`)
          and try again.
    """)


def setup(args):
    """
    Copied directly from detectron2/tools/train_net.py
    """
    cfg = get_cfg()

    ## Change here
    add_aldi_config(cfg)
    ## End change

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    """
    Runs evaluation and visualizes features.
    """
    cfg = setup(args)

    # load model
    model = ALDITrainer.build_model(cfg)
    ckpt = DetectionCheckpointerWithEMA(model, save_dir=cfg.OUTPUT_DIR)
    if cfg.EMA.ENABLED and cfg.EMA.LOAD_FROM_EMA_ON_START:
        ema = EMA(ALDITrainer.build_model(cfg), cfg.EMA.ALPHA)
        ckpt.add_checkpointable("ema", ema)
    ckpt.resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
    
    # feature map options
    pooling_method = F.avg_pool2d  # or F.max_pool2d

    # get source and target datasets
    if len(cfg.DATASETS.TEST) == 2:
        dataset_names = cfg.DATASETS.TEST
    elif len(cfg.DATASETS.TEST) == 1:
        assert len(cfg.DATASETS.TRAIN) == 1
        dataset_names = [cfg.DATASETS.TRAIN[0], cfg.DATASETS.TEST[0]]
    else:
        raise ValueError("Ambiguous which datasets represent source and target")

    image_level_features = {}
    proposal_level_features = {}
    for dataset_name in dataset_names:

        # keep track of image-level features
        image_level_features[dataset_name] = []
        backbone_hook_handle = model.backbone.register_forward_hook(lambda module, input, output: image_level_features[dataset_name].extend(pooling_method(output[sorted(output.keys())[-1]], kernel_size=output[sorted(output.keys())[-1]].shape[2:4])[..., 0, 0].detach().cpu().numpy()))
        
        # keep track of proposal-level features
        proposal_level_features[dataset_name] = []
        roi_heads_hook_handle = model.roi_heads.box_pooler.register_forward_hook(lambda module, input, output: proposal_level_features[dataset_name].extend(pooling_method(output, kernel_size=output.shape[2:4])[..., 0, 0].detach().cpu().numpy()))

        # iterate over single dataset
        data_loader = ALDITrainer.build_test_loader(cfg, dataset_name)
        inference_on_dataset(model, data_loader, evaluator=None)
        
        # clean up hooks
        backbone_hook_handle.remove()
        roi_heads_hook_handle.remove()

    # transform and visualize image-level features
    pca = PCA(n_components=2).fit(np.concatenate(list(image_level_features.values()), axis=0))
    image_level_features = { k: pca.transform(v) for k, v in image_level_features.items() }
    for dataset_name in dataset_names:
        plt.scatter(x=image_level_features[dataset_name][:, 0], y=image_level_features[dataset_name][:, 1], label=dataset_name, alpha=0.5, s=1)
    leg = plt.legend(markerscale=4)
    for lh in leg.legend_handles: 
        lh.set_alpha(1)
    plt.title(f"Image-level PCA (exp. var. {pca.explained_variance_ratio_.sum():.2f})")
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, f"feature_vis_image_pca.png"), bbox_inches="tight")
    plt.close("all")

    # transform and visualize proposal-level features
    pca = PCA(n_components=2).fit(np.concatenate(list(proposal_level_features.values()), axis=0))
    proposal_level_features = { k: pca.transform(v) for k, v in proposal_level_features.items() }
    for dataset_name in dataset_names:
        plt.scatter(x=proposal_level_features[dataset_name][:, 0], y=proposal_level_features[dataset_name][:, 1], label=dataset_name, alpha=0.1, s=1)
    leg = plt.legend(markerscale=4)
    for lh in leg.legend_handles: 
        lh.set_alpha(1)
    plt.title(f"Proposal-level PCA (exp. var. {pca.explained_variance_ratio_.sum():.2f})")
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, f"feature_vis_proposal_pca.png"), bbox_inches="tight")
    plt.close("all")

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        timeout=timedelta(minutes=1), # added for debugging
        args=(args,),
    )
