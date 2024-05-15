import copy
import torch
import numpy as np

from detectron2.structures import Instances, Boxes

from aldi.aug import WEAK_IMG_KEY
from aldi.dropin import DatasetMapper


class SaveWeakDatasetMapper(DatasetMapper):
    """
    DatasetMapper that retrieves the weakly augmented image from the aug_input object
    and saves it in the dataset_dict. See aug.SaveImgAug.
    """
    def _after_call(self, dataset_dict, aug_input):
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
            if WEAK_IMG_KEY in img:
                img["image"] = img[WEAK_IMG_KEY]
    labeled_strong = labeled if "labeled_strong" in batch_contents else None

    # unlike labeled data, we always return unlabeled_weak if *any* unlabeled data is requested
    # this allows us to do pseudo-labeling using the weakly augmented target data
    unlabeled_weak = None
    if ("unlabeled_weak" in batch_contents or "unlabeled_strong" in batch_contents) and unlabeled is not None:
        unlabeled_weak = copy.deepcopy(unlabeled)
        for img in unlabeled_weak:
            if WEAK_IMG_KEY in img:
                img["image"] = img[WEAK_IMG_KEY]
    unlabeled_strong = unlabeled if "unlabeled_strong" in batch_contents else None

    return labeled_weak, labeled_strong, unlabeled_weak, unlabeled_strong