from typing import Literal
import copy
import os
import torch
import numpy as np

from detectron2.structures import Instances, Boxes
from detectron2.data import detection_utils as utils, MetadataCatalog

from aug import WEAK_IMG_KEY
from dropin import DatasetMapper

TRANSLATED_IMG_KEY = "img_translated"

class UMTCapableDatasetMapper(DatasetMapper):
    """
    DatasetMapper that enables loading source-like and target-like
    samples corresponding to labeled target and unlabeled source samples.
    """
    def __init__(self, cfg, *args, dataset_type: Literal["source", "target"]=None, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.do_umt = cfg.MODEL.UMT.ENABLED
        if not self.do_umt:
            return
        if dataset_type is None:
            raise TypeError("dataset_type must be either 'source' or 'target'")
        assert len(cfg.DATASETS.TRAIN) == 1, "only the usage of a single labeled training dataset is currently implemented for UMT"
        assert len(cfg.DATASETS.UNLABELED) == 1, "only the usage of a single unlabeled training dataset is currently implemented for UMT"
        self.meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0] if dataset_type == "source" else cfg.DATASETS.UNLABELED[0])

    def _after_call(self, dataset_dict, aug_input, transforms):
        if not self.do_umt:
            return dataset_dict

        # load source-like or target-like samples for UMT
        dataset_dict = copy.deepcopy(dataset_dict)
        image_translated = utils.read_image(os.path.join(self.meta.translated_image_dir, os.path.relpath(dataset_dict["file_name"], self.meta.image_dir_prefix)), format=self.image_format)
        image_translated = transforms.apply_image(image_translated)
        dataset_dict[TRANSLATED_IMG_KEY] = torch.as_tensor(np.ascontiguousarray(image_translated.transpose(2, 0, 1)))

        return dataset_dict

class SaveWeakDatasetMapper(UMTCapableDatasetMapper):
    """
    DatasetMapper that retrieves the weakly augmented image from the aug_input object
    and saves it in the dataset_dict. See aug.SaveImgAug.
    """
    def _after_call(self, dataset_dict, aug_input, transforms):
        dataset_dict = super()._after_call(dataset_dict, aug_input, transforms)
        weak_img = getattr(aug_input, WEAK_IMG_KEY)
        dataset_dict[WEAK_IMG_KEY] = torch.as_tensor(np.ascontiguousarray(weak_img.transpose(2, 0, 1)))
        return dataset_dict

class UnlabeledDatasetMapper(SaveWeakDatasetMapper):
    def __call__(self, dataset_dict):
        dataset_dict = super().__call__(dataset_dict)

        # delete any gt boxes
        dataset_dict.pop("annotations", None)
        dataset_dict.pop("sem_seg_file_name", None)
        dataset_dict['instances'] = Instances(dataset_dict['instances'].image_size, gt_boxes=Boxes([]), 
                                              gt_classes=torch.tensor([], dtype=torch.int64))
        return dataset_dict

class TwoDataloaders:
    class NoneIterator:
        def __next__(self):
            return None
        
    def __init__(self, loader0, loader1):
        self.loader0 = TwoDataloaders.NoneIterator() if loader0 is None else iter(loader0)
        self.loader1 = TwoDataloaders.NoneIterator() if loader1 is None else iter(loader1)
    
    def __iter__(self):
        while True:
            yield (next(self.loader0), next(self.loader1))

class WeakStrongDataloader:
    def __init__(self, labeled_loader, unlabeled_loader, batch_contents=("labeled_weak", "labeled_strong", "unlabeled_strong")):
        self.loader = TwoDataloaders(labeled_loader, unlabeled_loader)
        self.batch_contents = batch_contents
    
    def __iter__(self):
        for batch in self.loader:
            yield unpack_data_weak_strong(*batch, batch_contents=self.batch_contents)

    def __len__(self):
        return len(self.loader)

def unpack_data_weak_strong(labeled, unlabeled, batch_contents=("labeled_weak", "labeled_strong", "unlabeled_strong")):
    """
    Postprocess data from a SaveWeakDatasetMapper to expose both weakly and strongly augmented images.
    Return: (labeled_weak, labeled_strong, unlabeled_weak, unlabeled_strong)
    """
    labeled_weak = None
    if "labeled_weak" in batch_contents and labeled is not None:
        labeled_weak = copy.deepcopy(labeled)
        for img in labeled_weak:
            if TRANSLATED_IMG_KEY in img:
                img["image"] = img[TRANSLATED_IMG_KEY]
            elif WEAK_IMG_KEY in img:
                img["image"] = img[WEAK_IMG_KEY]
    labeled_strong = labeled if "labeled_strong" in batch_contents else None

    # unlike labeled data, we always return unlabeled_weak if *any* unlabeled data is requested
    # this allows us to do pseudo-labeling using the weakly augmented target data
    unlabeled_weak = None
    if ("unlabeled_weak" in batch_contents or "unlabeled_strong" in batch_contents) and unlabeled is not None:
        unlabeled_weak = copy.deepcopy(unlabeled)
        for img in unlabeled_weak:
            if TRANSLATED_IMG_KEY in img:
                img["image"] = img[TRANSLATED_IMG_KEY]
            elif WEAK_IMG_KEY in img:
                img["image"] = img[WEAK_IMG_KEY]
    unlabeled_strong = unlabeled if "unlabeled_strong" in batch_contents else None

    return labeled_weak, labeled_strong, unlabeled_weak, unlabeled_strong