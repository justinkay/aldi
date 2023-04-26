import os
import torch
import copy
import weakref

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from detectron2.modeling.meta_arch.build import build_model
from detectron2.data.build import build_detection_train_loader, get_detection_dataset_dicts
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import hooks, BestCheckpointer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils import comm
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from aug import WEAK_IMG, get_augs
from dataloader import UnlabeledDatasetMapper, PrefetchableConcatDataloaders
from meanteacher import EmaRCNN
from pseudolabels import add_label, process_pseudo_label


# save intermediate objects during execution for debugging
DEBUG = False

class DATrainer(DefaultTrainer):
    """
    Main idea:
        We are just "training" the student.
            -> "Step" in the training loop refers to a student step.
            Problem here is we may want different losses for labeled/unlabeld data
        We may be updating a teacher model.
        We may also be "training" other networks.
        But the Trainer is training the student.

    Assumption:
        If doing adaptation, student is already "burned in" by another trainer.
    """
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """Just do COCO Evaluation."""
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return DatasetEvaluators([COCOEvaluator(dataset_name, output_dir=output_folder)])
    
    def __init__(self, cfg):
        super().__init__(cfg)

        # EMA of student
        if cfg.EMA.ENABLED:
            self.ema = EmaRCNN(build_model(cfg), cfg.EMA.ALPHA)
            self.ema_checkpointer = DetectionCheckpointer(
                self.ema.model,
                cfg.OUTPUT_DIR,
                trainer=weakref.proxy(self),
        )

        # build hooks after EMA is initialized
        self.register_hooks(self.build_hooks_late())

        # Debugging variables
        if DEBUG:
            self._last_labeled, self._last_unlabeled, self._last_pseudo = None, None, None


    @classmethod
    def build_train_loader(cls, cfg):
        total_batch_size = cfg.SOLVER.IMS_PER_BATCH
        labeled_bs, unlabeled_bs = ( int(r * total_batch_size / sum(cfg.DATASETS.LABELED_UNLABELED_RATIO)) for r in cfg.DATASETS.LABELED_UNLABELED_RATIO )
        loaders = []

        if labeled_bs > 0 and len(cfg.DATASETS.TRAIN):
            labeled_loader = build_detection_train_loader(get_detection_dataset_dicts(cfg.DATASETS.TRAIN, filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS), 
                mapper=DatasetMapper(cfg, is_train=True, augmentations=get_augs(cfg, labeled=True)),
                num_workers=cfg.DATALOADER.NUM_WORKERS, 
                total_batch_size=labeled_bs)
            loaders.append(labeled_loader)

        # if we are utilizing unlabeled data, add it to the dataloader
        if unlabeled_bs > 0 and len(cfg.DATASETS.UNLABELED):
            unlabeled_loader = build_detection_train_loader(get_detection_dataset_dicts(cfg.DATASETS.UNLABELED, filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS), 
                mapper=UnlabeledDatasetMapper(cfg, is_train=True, augmentations=get_augs(cfg, labeled=True)),
                num_workers=cfg.DATALOADER.NUM_WORKERS, # should we do this? two dataloaders...
                total_batch_size=unlabeled_bs)
            loaders.append(unlabeled_loader)

        return PrefetchableConcatDataloaders(loaders)

    def build_hooks(self):
        """Disable hooks in superclass initialization; build them after EMA model is built."""
        return []
    
    def build_hooks_late(self):
        """Add hooks for saving EMA model."""
        ret = super().build_hooks()

        # add checkpoint and eval for EMA model if enabled
        if self.cfg.EMA.ENABLED:
            if comm.is_main_process():
                ret.insert(-1, # before the PeriodicWriter; see DefaultTrainer.build_hooks()
                           hooks.PeriodicCheckpointer(self.ema_checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD, file_prefix="model_teacher")) 

            def test_and_save_results_ema():
                self._last_eval_results = self.test(self.cfg, self.ema.model)
                return self._last_eval_results

            # Do evaluation after checkpointer, because then if it fails,
            # we can use the saved checkpoint to debug.
            eval_hook = hooks.EvalHook(self.cfg.TEST.EVAL_PERIOD, test_and_save_results_ema)
            if comm.is_main_process():
                ret.insert(-1, eval_hook) # again, before PeriodicWriter if in main process
            else:
                ret.append(eval_hook)

        # add a hook to save the best (teacher, if EMA enabled) checkpoint to model_best.pth
        if comm.is_main_process():
            for test_set in self.cfg.DATASETS.TEST:
                ret.insert(-1, BestCheckpointer(self.cfg.TEST.EVAL_PERIOD, self.checkpointer, f"{test_set}/bbox/AP50", "max", file_prefix=f"{test_set}_model_best"))
        return ret

    def run_step(self):
        """Remember that self._trainer is the student trainer."""
        
        # Prefetch dataloader batch so we can add pseudo labels from teacher as needed
        data = self._trainer.data_loader.prefetch_batch()
        if len(data) == 1:
            labeled, unlabeled = data[0], None
        elif len(data) == 2:
            labeled, unlabeled = data
        else:
            raise ValueError("Unsupported number of dataloaders")
        
        if DEBUG:
            self._last_labeled = copy.deepcopy(labeled)
            self._last_unlabeled = copy.deepcopy(unlabeled)

        # EMA update
        if self.cfg.EMA.ENABLED:
            self.ema.update_weights(self.model, self.iter)

        # Teacher-student self-training
        if self.cfg.DOMAIN_ADAPT.TEACHER.ENABLED:

            # img["image"] currently contains strongly augmented images;
            # we want to generate pseudo labels using the weakly augmented images
            for img in unlabeled:
                img["original_image"] = img["image"]
                img["image"] = img[WEAK_IMG]

            if DEBUG:
                self._last_unlabeled_weak = copy.deepcopy(unlabeled)

            with torch.no_grad():
                # run teacher on weakly augmented data
                self.ema.model.eval()
                teacher_preds = self.ema.model.inference(unlabeled, do_postprocess=False)
                
                if DEBUG:
                    self._last_teacher_preds = copy.deepcopy(teacher_preds)

                # postprocess pseudo labels (thresholding)
                teacher_preds, _ = process_pseudo_label(teacher_preds, self.cfg.DOMAIN_ADAPT.TEACHER.THRESHOLD, 
                                                             "roih", self.cfg.DOMAIN_ADAPT.TEACHER.PSEUDO_LABEL_METHOD)
                
                # add pseudo labels back as "ground truth"
                unlabeled = add_label(unlabeled, teacher_preds)

            # restore strongly augmented images for student
            for img in unlabeled:
                img["image"] = img["original_image"]

        if DEBUG:
            self._last_pseudo = copy.deepcopy(unlabeled)
            self._last_prefetched = self._trainer.data_loader.prefetched_data

        # now call student.run_step as normal
        self._trainer.iter = self.iter
        self._trainer.run_step()