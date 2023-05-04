import torch
import numpy as np

from detectron2.structures import Instances, Boxes

from aug import WEAK_IMG_KEY
from dropin import DatasetMapper

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