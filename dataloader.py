import copy
import torch
import numpy as np

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

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
        dataset_dict.pop("annotations", None)
        dataset_dict.pop("sem_seg_file_name", None)
        return dataset_dict

class PrefetchableConcatDataloaders:
    """
    Multiple dataloaders whose batches are concatenated.
    They can also be "prefetched" so that the next batch is already loaded, allowing
    use and modification of data before it hits the default Detectron2 training logic.
    (E.g. the batch can be modified with weak/strong augmentation and pseudo labeling)
    """
    def __init__(self, loaders: list):
        self.iters = [iter(loader) for loader in loaders]
        self.prefetched_data = None
    
    def __iter__(self):
        while True:
            if self.prefetched_data is None:
                outputs = self._get_next_batch()
            else:
                outputs = self.prefetched_data
                self.clear_prefetch()
            yield outputs

    def prefetch_batch(self):
        assert self.prefetched_data is None, "Prefetched data already exists"
        self.prefetched_data = self._get_next_batch()
        return self.prefetched_data

    def _get_next_batch(self):
        return [next(it) for it in self.iters]

    def clear_prefetch(self):
        self.prefetched_data = None