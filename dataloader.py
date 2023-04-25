import copy
import torch
import numpy as np

from detectron2.data import DatasetMapper
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from aug import WEAK_IMG

class SaveWeakDatasetMapper(DatasetMapper):
    def __call__(self, dataset_dict):
        """
        DatasetMapper that retrieves the weakly augmented image from the aug_input object
        and saves it in the dataset_dict. See aug.SaveImgAug.
        Would be great to avoid copy and pasting this method but haven't found a way yet.
        """

        ### Direct copy from DatasetMapper ###
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None
        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )
        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict
        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)
        ### End direct copy from DatasetMapper ###

        # Save weakly augmented image in dataset_dict
        weak_img = getattr(aug_input, WEAK_IMG)
        dataset_dict[WEAK_IMG] = torch.as_tensor(np.ascontiguousarray(weak_img.transpose(2, 0, 1)))

        return dataset_dict

class UnlabeledDatasetMapper(SaveWeakDatasetMapper):
    def __call__(self, dataset_dict):
        dataset_dict = super().__call__(dataset_dict)
        dataset_dict.pop("annotations", None)
        dataset_dict.pop("sem_seg_file_name", None)
        return dataset_dict

class PrefetchableConcatDataloaders:
    """
    Two dataloaders, one labeled, one unlabeled, whose batches are concatenated.
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
            yield [ x for o in outputs for x in o ]

    def prefetch_batch(self):
        assert self.prefetched_data is None, "Prefetched data already exists"
        self.prefetched_data = self._get_next_batch()
        return self.prefetched_data

    def _get_next_batch(self):
        return [next(it) for it in self.iters]

    def clear_prefetch(self):
        self.prefetched_data = None