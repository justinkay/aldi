from detectron2.data import DatasetMapper

class UnlabeledDatasetMapper(DatasetMapper):
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
    def __init__(self, labeled_loader, unlabeled_loader):
        self.labeled_iter = iter(labeled_loader)
        self.unlabeled_iter = iter(unlabeled_loader)
        self.prefetched_data = None
    
    def __iter__(self):
        while True:
            if self.prefetched_data is None:
                labeled, unlabeled = self._get_next_batch()
            else:
                labeled, unlabeled = self.prefetched_data
                self.clear_prefetch()
            yield labeled + unlabeled

    def prefetch_batch(self):
        assert self.prefetched_data is None, "Prefetched data already exists"
        self.prefetched_data = self._get_next_batch()
        return self.prefetched_data

    def _get_next_batch(self):
        return next(self.labeled_iter), next(self.unlabeled_iter)

    def clear_prefetch(self):
        self.prefetched_data = None