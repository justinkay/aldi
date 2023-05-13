import os
import torch
import copy
import logging
from torch.nn.parallel import DistributedDataParallel as DDP

from detectron2.data.build import build_detection_train_loader, get_detection_dataset_dicts
from detectron2.engine import hooks, BestCheckpointer
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from detectron2.modeling.meta_arch.build import build_model
from detectron2.solver import build_optimizer
from detectron2.utils.events import get_event_storage
from detectron2.utils import comm

from aug import WEAK_IMG_KEY, get_augs
from dropin import DefaultTrainer, AMPTrainer, SimpleTrainer
from dataloader import SaveWeakDatasetMapper, UnlabeledDatasetMapper, TwoDataloaders
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
          data: a tuple of outputs from two dataloaders (labeled, unlabeled)
          teacher: a teacher model to use for generating pseudo-labels, or None to disable pseudo-labeling
          threshold: the threshold to use for pseudo-labeling if using a threshold based method
          method: the method to use for pseudo-labeling, {"thresholding", }
          trainer: (optional) used for debugging purposes only
          include_weak_in_batch: if True, add weakly-augmented source images to the batch
     """
     labeled, unlabeled = data
     loss_dict = {}

     # get weakly augmented version of batch
     # TODO: not intuitive that labeled might already be weakly-augmented
     labeled_weak = copy.deepcopy(labeled)
     for img in labeled_weak:
          img["image"] = img[WEAK_IMG_KEY]
     unlabeled_weak = copy.deepcopy(unlabeled)
     for img in unlabeled_weak:
          img["image"] = img[WEAK_IMG_KEY]

     #### Weakly augmented source imagery
     #### (Used for normal training and/or domain alignment)
     _model = model.module if type(model) == DDP else model
     do_sada = labeled is not None and _model.da_heads is not None # TODO
     do_weak = include_weak_in_batch or do_sada # TODO
     if do_weak:
          loss_weak = model(labeled_weak, do_sada=do_sada)
          for k, v in loss_weak.items():
               if include_weak_in_batch or (do_sada and "_da_" in k):
                    loss_dict[f"{k}_source_weak"] = v

     #### Weakly augmented target imagery
     #### (Only used for domain alignment)
     if do_sada:
          loss_sada_target = model(unlabeled_weak, labeled=False, do_sada=True)
          for k, v in loss_sada_target.items():
               if "_da_" in k:
                    loss_dict[f"{k}_target_weak"] = v

     #### Strongly augmented source imagery
     #### These values are kept as the standard loss names (e.g. "loss_cls")
     do_strong = labeled is not None # TODO
     if do_strong:
          loss_dict.update(model(labeled, do_sada=False))

     if DEBUG and trainer is not None:
          trainer._last_labeled = copy.deepcopy(labeled)
          trainer._last_unlabeled = copy.deepcopy(unlabeled)

     #### Pseudo-labeled target imagery
     if unlabeled is not None:
          # are we using an (EMA) teacher model to generate pseudo-labels?
          # if not, the (student) model generates its own pseudo-labels
          # TODO: this would probably be cleaner to just pass in the student model as teacher
          #         and track whether teacher was in train mode before (in that case, revert after inference)
          training_with_student = teacher is None
          if training_with_student:
               teacher = model.module if type(model) == DDP else model

          with torch.no_grad():
               if DEBUG and trainer is not None:
                    trainer._last_unlabeled_before_teacher = copy.deepcopy(unlabeled)
                    
               # run teacher on weakly augmented data
               # do_postprocess=False to disable transforming outputs back into original image space
               teacher.eval()
               teacher_preds = teacher.inference(unlabeled_weak, do_postprocess=False)
               if training_with_student: teacher.train()

               # postprocess pseudo labels (thresholding)
               teacher_preds, _ = process_pseudo_label(teacher_preds, threshold, "roih", method)
               
               # add pseudo labels back as "ground truth"
               unlabeled = add_label(unlabeled, teacher_preds)

          if DEBUG and trainer is not None:
               trainer._last_unlabeled_after_teacher = copy.deepcopy(unlabeled)
               with torch.no_grad():
                    model.eval()
                    if type(model) == DDP:
                         trainer._last_student_preds = model.module.inference(unlabeled, do_postprocess=False)
                    else:
                         trainer._last_student_preds = model.inference(unlabeled, do_postprocess=False)
                    model.train()

          losses_pseudolabeled = model(unlabeled, labeled=False, do_sada=False)
          for k, v in losses_pseudolabeled.items():
               loss_dict[k + "_pseudo"] = v
     
     # scale the loss to account for the gradient accumulation we've done
     # TODO: this does not include SADA - how should the DA loss be modified?
     # TODO: loss_box_reg is actually normalized by number of gt boxes, so this isn't a perfect solution
     num_grad_accum_steps = int(labeled is not None) + int(unlabeled is not None) + int(include_weak_in_batch)
     for k, v in loss_dict.items():
          loss_dict[k] = v / num_grad_accum_steps

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
          # build EMA model if applicable
          trainer.ema = None
          if cfg.EMA.ENABLED:
               trainer.ema = EMA(build_model(cfg), cfg.EMA.ALPHA)
          return trainer
     
     def _create_checkpointer(self, model, cfg):
          checkpointer = super(DATrainer, self)._create_checkpointer(model, cfg)
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
          ret = super(DATrainer, self).build_hooks()

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
     def build_optimizer(cls, cfg, model):
          """
          Change the learning rate to account for the fact that we are doing gradient accumulation in order
          to run multiple batches of labeled and unlabeled data each training step.
          """
          logger = logging.getLogger("detectron2")
          num_grad_accum = (sum(cfg.DATASETS.LABELED_UNLABELED_RATIO) + int(cfg.DATASETS.INCLUDE_WEAK_IN_BATCH))
          effective_batch_size = cfg.SOLVER.IMS_PER_BATCH * num_grad_accum
          logger.info(f"Effective batch size is {effective_batch_size} due to {num_grad_accum} gradient accumulation steps.")
          lr_scale = effective_batch_size / cfg.SOLVER.IMS_PER_BATCH
          logger.info(f"Scaling LR from {cfg.SOLVER.BASE_LR} to {lr_scale * cfg.SOLVER.BASE_LR}.")
          cfg.defrost()
          cfg.SOLVER.BASE_LR = lr_scale * cfg.SOLVER.BASE_LR
          cfg.freeze()
          return super(DATrainer, cls).build_optimizer(cfg, model)

     @classmethod
     def build_train_loader(cls, cfg):
          total_batch_size = cfg.SOLVER.IMS_PER_BATCH
          labeled_bs, unlabeled_bs = ( int(r * total_batch_size / max(cfg.DATASETS.LABELED_UNLABELED_RATIO)) for r in cfg.DATASETS.LABELED_UNLABELED_RATIO )

          # create labeled dataloader
          labeled_loader = None
          if labeled_bs > 0 and len(cfg.DATASETS.TRAIN):
               labeled_loader = build_detection_train_loader(get_detection_dataset_dicts(cfg.DATASETS.TRAIN, filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS), 
                    mapper=SaveWeakDatasetMapper(cfg, is_train=True, augmentations=get_augs(cfg, labeled=True)),
                    num_workers=cfg.DATALOADER.NUM_WORKERS, 
                    total_batch_size=labeled_bs)

          # create unlabeled dataloader
          unlabeled_loader = None
          if unlabeled_bs > 0 and len(cfg.DATASETS.UNLABELED):
               unlabeled_loader = build_detection_train_loader(get_detection_dataset_dicts(cfg.DATASETS.UNLABELED, filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS), 
                    mapper=UnlabeledDatasetMapper(cfg, is_train=True, augmentations=get_augs(cfg, labeled=False)),
                    num_workers=cfg.DATALOADER.NUM_WORKERS, # should we do this? two dataloaders...
                    total_batch_size=unlabeled_bs)

          return TwoDataloaders(labeled_loader, unlabeled_loader)
     
     def before_step(self):
          super(DATrainer, self).before_step()
          if self.cfg.EMA.ENABLED:
               self._trainer.ema.update_weights(self._trainer.model, self.iter)