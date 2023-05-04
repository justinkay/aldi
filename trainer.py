import os
import logging
import weakref
import torch
import copy
from torch.nn.parallel import DistributedDataParallel as DDP

from detectron2.data.build import build_detection_train_loader, get_detection_dataset_dicts
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import HookBase, hooks, BestCheckpointer
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from detectron2.modeling.meta_arch.build import build_model
from detectron2.utils.events import get_event_storage
from detectron2.utils import comm

from aug import WEAK_IMG_KEY, get_augs
from dropin import DefaultTrainer, AMPTrainer, SimpleTrainer
from dataloader import SaveWeakDatasetMapper, UnlabeledDatasetMapper, PrefetchableDataloaders
from ema import EMA
from pseudolabels import add_label, process_pseudo_label


DEBUG = False

def run_model_labeled_unlabeled(model, data, teacher=None, threshold=0.8, method="thresholding", trainer=None,
                                   include_weak_in_batch=False):
     """
     Main logic of running Mean Teacher style training for one iteration.
     - If no unlabeled data is supplied, run a training step as usual.
     - If unlabeled data is supplied, uses gradient accumulation to run the model on both labeled and 
     unlabeled data in the same step. This approach is taken (vs., e.g., concatenating the labeled and 
     unlabeled data into a single batch) to allow for easier customization of training losses.

     Assumes any images to be pseudo-labeled have a weakly augmented version stored in the dataset dict
     under the key aug.WEAK_IMG_KEY.

     Args:
          data: either output from a single (labeled) dataloader or a tuple of outputs from two 
                dataloaders (labeled, unlabeled)
          teacher: a teacher model to use for generating pseudo-labels, or None to disable pseudo-labeling
          threshold: the threshold to use for pseudo-labeling if using a threshold based method
          method: the method to use for pseudo-labeling, {"thresholding", }
          trainer: (optional) used for debugging purposes only
          include_weak_in_batch: if True, add weakly-augmented source images to the batch
     """
     if len(data) == 1:
          labeled, unlabeled = data[0], None
     elif len(data) == 2:
          labeled, unlabeled = data
     else:
          raise ValueError(f"Unsupported number of dataloaders, {len(data)}")

     loss_dict = {}

     # run model on labeled data per usual
     # these values are kept as the standard loss names (e.g. "loss_cls")
     loss_dict.update(model(labeled))

     if DEBUG and trainer is not None:
          trainer._last_labeled = copy.deepcopy(labeled)
          trainer._last_unlabeled = copy.deepcopy(unlabeled)

     # run model on unlabeled data
     if unlabeled is not None:
          # are we using an (EMA) teacher model to generate pseudo-labels?
          # if not, the (student) model generates its own pseudo-labels
          training_with_student = teacher is None
          if training_with_student:
               teacher = model.module if type(model) == DDP else model

          # img["image"] currently contains strongly augmented images;
          # we want to generate pseudo-labels using the weakly augmented images
          for img in unlabeled:
               img["original_image"] = img["image"]
               img["image"] = img[WEAK_IMG_KEY]

          with torch.no_grad():
               if DEBUG and trainer is not None:
                    trainer._last_unlabeled_before_teacher = copy.deepcopy(unlabeled)
                    
               # run teacher on weakly augmented data
               # do_postprocess=False to disable transforming outputs back into original image space
               teacher.eval()
               teacher_preds = teacher.inference(unlabeled, do_postprocess=False)
               if training_with_student: teacher.train()

               # postprocess pseudo labels (thresholding)
               teacher_preds, _ = process_pseudo_label(teacher_preds, threshold, "roih", method)
               
               # add pseudo labels back as "ground truth"
               unlabeled = add_label(unlabeled, teacher_preds)

               # restore strongly augmented images for student
               for img in unlabeled:
                    img["image"] = img["original_image"]

          if DEBUG and trainer is not None:
               trainer._last_unlabeled_after_teacher = copy.deepcopy(unlabeled)
               with torch.no_grad():
                    model.eval()
                    if type(model) == DDP:
                         trainer._last_student_preds = model.module.inference(unlabeled, do_postprocess=False)
                    else:
                         trainer._last_student_preds = model.inference(unlabeled, do_postprocess=False)
                    model.train()

          losses_unlabeled = model(unlabeled, labeled=False)
          for k, v in losses_unlabeled.items():
               loss_dict[k + "_pseudo"] = v

     # run model on weakly-augmented labeled data if desired
     if include_weak_in_batch:
          for img in labeled:
               img["image"] = img[WEAK_IMG_KEY]
          loss_weak = model(labeled)
          for k, v in loss_weak.items():
               loss_dict[f"{k}_weak"] = v
               
     return loss_dict

class DAAMPTrainer(AMPTrainer):
     def __init__(self, model, data_loader, optimizer, pseudo_label_method, pseudo_label_thresh, include_weak_in_batch=False):
          super().__init__(model, data_loader, optimizer)
          self.pseudo_label_method = pseudo_label_method
          self.pseudo_label_thresh = pseudo_label_thresh
          self.include_weak_in_batch = include_weak_in_batch

     def run_model(self, data):
          teacher = self.ema.model if self.ema is not None else None
          return run_model_labeled_unlabeled(self.model, data, teacher=teacher, 
                                             threshold=self.pseudo_label_thresh, 
                                             method=self.pseudo_label_method, trainer=self,
                                             include_weak_in_batch=self.include_weak_in_batch)
     
class DASimpleTrainer(SimpleTrainer):
     def __init__(self, model, data_loader, optimizer, pseudo_label_method, pseudo_label_thresh, include_weak_in_batch=False):
          super().__init__(model, data_loader, optimizer)
          self.pseudo_label_method = pseudo_label_method
          self.pseudo_label_thresh = pseudo_label_thresh
          self.include_weak_in_batch = include_weak_in_batch

     def run_model(self, data):
          teacher = self.ema.model if self.ema is not None else None
          return run_model_labeled_unlabeled(self.model, data, teacher=teacher, 
                                             threshold=self.pseudo_label_thresh, 
                                             method=self.pseudo_label_method, trainer=self,
                                             include_weak_in_batch=self.include_weak_in_batch)
     
class DATrainer(DefaultTrainer):
     def _create_trainer(self, cfg, model, data_loader, optimizer):
          trainer = (DAAMPTrainer if cfg.SOLVER.AMP.ENABLED else DASimpleTrainer)(model, data_loader, optimizer, cfg.DOMAIN_ADAPT.TEACHER.PSEUDO_LABEL_METHOD,
                              cfg.DOMAIN_ADAPT.TEACHER.THRESHOLD, cfg.DATASETS.INCLUDE_WEAK_IN_BATCH)
          trainer.ema = None
          if cfg.EMA.ENABLED:
               trainer.ema = EMA(build_model(cfg), cfg.EMA.ALPHA)
          return trainer
     
     def _create_checkpointer(self, model, cfg):
          checkpointer = super()._create_checkpointer(model, cfg)
          if cfg.EMA.ENABLED:
               checkpointer.add_checkpointable("ema", self._trainer.ema.model)
          return checkpointer

     @classmethod
     def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """Just do COCO Evaluation."""
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return DatasetEvaluators([COCOEvaluator(dataset_name, output_dir=output_folder)])

     def build_hooks(self):
          ret = super().build_hooks()

          # add hooks to evaluate/save teacher model if applicable
          if self.cfg.EMA.ENABLED:
               def test_and_save_results_ema():
                    self._last_eval_results = self.test(self.cfg, self._trainer.ema.model)
                    return self._last_eval_results
               eval_hook = hooks.EvalHook(self.cfg.TEST.EVAL_PERIOD, test_and_save_results_ema)
               if comm.is_main_process():
                    ret.insert(-1, eval_hook) # before PeriodicWriter if in main process
               else:
                    ret.append(eval_hook)

          # add a hook to save the best (teacher, if EMA enabled) checkpoint to model_best.pth
          if comm.is_main_process():
               for test_set in self.cfg.DATASETS.TEST:
                    ret.insert(-1, BestCheckpointer(self.cfg.TEST.EVAL_PERIOD, self.checkpointer, #ema_checkpointer if ema_checkpointer is not None else self.checkpointer, 
                                                    f"{test_set}/bbox/AP50", "max", file_prefix=f"{test_set}_model_best"))

          return ret
     
     @classmethod
     def build_train_loader(cls, cfg):
          total_batch_size = cfg.SOLVER.IMS_PER_BATCH
          labeled_bs, unlabeled_bs = ( int(r * total_batch_size / max(cfg.DATASETS.LABELED_UNLABELED_RATIO)) for r in cfg.DATASETS.LABELED_UNLABELED_RATIO )
          loaders = []

          # create labeled dataloader
          if labeled_bs > 0 and len(cfg.DATASETS.TRAIN):
               labeled_loader = build_detection_train_loader(get_detection_dataset_dicts(cfg.DATASETS.TRAIN, filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS), 
                    mapper=SaveWeakDatasetMapper(cfg, is_train=True, augmentations=get_augs(cfg, labeled=True)),
                    num_workers=cfg.DATALOADER.NUM_WORKERS, 
                    total_batch_size=labeled_bs)
               loaders.append(labeled_loader)

          # create unlabeled dataloader
          if unlabeled_bs > 0 and len(cfg.DATASETS.UNLABELED):
               unlabeled_loader = build_detection_train_loader(get_detection_dataset_dicts(cfg.DATASETS.UNLABELED, filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS), 
                    mapper=UnlabeledDatasetMapper(cfg, is_train=True, augmentations=get_augs(cfg, labeled=False)),
                    num_workers=cfg.DATALOADER.NUM_WORKERS, # should we do this? two dataloaders...
                    total_batch_size=unlabeled_bs)
               loaders.append(unlabeled_loader)

          return PrefetchableDataloaders(loaders)
     
     def before_step(self):
          super().before_step()
          if self.cfg.EMA.ENABLED:
               self._trainer.ema.update_weights(self._trainer.model, self.iter)