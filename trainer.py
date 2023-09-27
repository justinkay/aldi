import os
import torch
import copy
import logging
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, always_wrap_policy

from detectron2.data.build import build_detection_train_loader, get_detection_dataset_dicts
from detectron2.engine import hooks, BestCheckpointer
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from detectron2.modeling.meta_arch.build import build_model
from detectron2.solver import build_optimizer
from detectron2.utils.events import get_event_storage
from detectron2.utils import comm

from aug import WEAK_IMG_KEY, get_augs
from backbone import get_adamw_optim, get_swinb_optim
from dropin import DefaultTrainer, AMPTrainer, SimpleTrainer
from dataloader import SaveWeakDatasetMapper, UnlabeledDatasetMapper, WeakStrongDataloader
from ema import EMA
from pseudolabeler import PseudoLabeler


def visualize_batch(labeled_weak, labeled_strong, unlabeled_weak, unlabeled_strong, max_rows=-1, umt_labels=False):
     """Helper method to visualize an entire or parts of a batch."""
     
     assert len(labeled_weak) == len(labeled_strong) == len(unlabeled_weak) == len(unlabeled_strong)
     
     from detectron2.utils.visualizer import Visualizer
     import matplotlib.pyplot as plt
     from mpl_toolkits.axes_grid1 import ImageGrid
     
     labels = ["target-like", "source", "source-like", "target"] if umt_labels else ["labeled weak", "labeled strong", "unlabeled weak", "unlabeled strong"]
     
     fig = plt.figure(figsize=(20., 20.))
     grid = ImageGrid(fig, 111, 
          nrows_ncols=(len(labeled_weak) if max_rows < 0 else min(max_rows, len(labeled_weak)), 4),
          axes_pad=0.1,
     )
     grid_iter = iter(grid)
     
     for row_idx, (lw, ls, uw, us) in enumerate(zip(labeled_weak, labeled_strong, unlabeled_weak, unlabeled_strong)):
          if max_rows >= 0 and row_idx >= max_rows:
               break
          for e, label in zip([lw, ls, uw, us], labels):
               ax = next(grid_iter)
               ax.imshow(Visualizer(e["image"].numpy().transpose(1, 2, 0)[..., ::-1]).draw_dataset_dict(e).get_image())
               ax.tick_params(
                    axis="both",
                    which="both",
                    left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)
               if row_idx == 0:
                    ax.set_title(label)
     return fig


DEBUG = True
debug_dict = {}

def run_model_labeled_unlabeled(trainer, labeled_weak, labeled_strong, unlabeled_weak, unlabeled_strong):
     """
     Main logic of running Mean Teacher style training for one iteration.
     - If no unlabeled data is supplied, run a training step as usual.
     - If unlabeled data is supplied, use teacher to create pseudo-labels
     Args:
          backward_at_end (bool): If True, losses are summed and returned, persisting the entire computation graph.
               This is memory intensive because we do gradient accumulation, but is faster.
               If False, call backward() throughout this method any time data is passed to the model. 
               This is slower, but uses less memory, allowing for larger batch sizes and training larger models that 
               would OOM with backward_at_end=True. Usually, the larger batch size also makes up for the slowdown.
     """
     model = trainer.model
     pseudo_labeler = trainer.pseudo_labeler
     backward_at_end = trainer.backward_at_end

     _model = model.module if type(model) == DDP else model
     do_sada = _model.sada_heads is not None
     do_weak = labeled_weak is not None
     do_strong = labeled_strong is not None
     do_unlabeled = unlabeled_weak is not None and pseudo_labeler is not None
     num_grad_accum_steps = int(do_weak) + int(do_strong) + int(do_unlabeled)

     if DEBUG:
          debug_dict['last_labeled_weak'] = copy.deepcopy(labeled_weak)
          debug_dict['last_labeled_strong'] = copy.deepcopy(labeled_strong)
          debug_dict['last_unlabeled_weak'] = copy.deepcopy(unlabeled_weak)
          debug_dict['last_unlabeled_strong'] = copy.deepcopy(unlabeled_strong)

     loss_dict = {}
     def add_to_loss_dict(losses, suffix, key_conditional=lambda k: True):
          """Helper method to add losses to loss_dict.
          Args:
               losses (dict): Dict of losses to add to loss_dict
               suffix (str): Suffix to add to each key in losses
               key_conditional (func): Function that takes a key and returns True/False whether to add it to loss_dict
          """
          for k, v in losses.items():
               if key_conditional(k):
                    v /= num_grad_accum_steps
                    loss_dict[f"{k}_{suffix}"] = v if backward_at_end else v.detach()

     def maybe_do_backward(losses, key_conditional=lambda k: True):
          """Helper method to do backward pass if not doing it at the end."""
          if not backward_at_end:
               losses = { k: v * 0 if not key_conditional(k) else v for k, v in losses.items() }
               trainer.do_backward(sum(losses.values()) / num_grad_accum_steps, override=True)

     # if DEBUG:
     #      # for debugging, visualize the contents of this batch
     #      visualize_batch(labeled_weak, labeled_strong, unlabeled_weak, unlabeled_strong, max_rows=3, umt_labels=True).savefig("batch_vis.png", bbox_inches="tight", transparent=True)

     #### Weakly-augmented source imagery (Used for normal training and/or domain alignment)
     if do_weak or do_sada:
          # Added try/catch for debugging Probabilistic Teacher - can hopefully remove later
          try:
               loss_weak = model(labeled_weak, do_sada=do_sada)
               maybe_do_backward(loss_weak, lambda k: do_weak or (do_sada and "_da_" in k))
               add_to_loss_dict(loss_weak, "source_weak", lambda k: do_weak or (do_sada and "_da_" in k))
          except FloatingPointError as e:
               print("Floating point error in weak forward pass. Skipping batch.")
               torch.save(labeled_weak, "labeled_weak_bad_batch.pt")
               return {"bad_loss": torch.tensor(0, device="cuda")}
     
     #### Weakly-augmented target imagery (Only used for domain alignment)
     if do_sada:
          loss_sada_target = model(unlabeled_weak, labeled=False, do_sada=True)
          maybe_do_backward(loss_sada_target, lambda k: "_da_" in k)
          add_to_loss_dict(loss_sada_target, "target_weak", lambda k: "_da_" in k)

     #### Strongly-augmented source imagery (Used for normal training)
     if do_strong:
          loss_strong = model(labeled_strong, do_sada=False)
          maybe_do_backward(loss_strong)
          add_to_loss_dict(loss_strong, "source_strong")

     #### Target imagery (Used for pseudo-labeling)
     if do_unlabeled:
          pseudolabeled_data = pseudo_labeler(unlabeled_weak, unlabeled_strong)
          loss_pseudolabeled = model(pseudolabeled_data, labeled=False, do_sada=False)
          maybe_do_backward(loss_pseudolabeled)
          add_to_loss_dict(loss_pseudolabeled, "target_pseudolabeled")
          if DEBUG: 
               debug_dict['last_pseudolabeled'] = copy.deepcopy(pseudolabeled_data)
     
     return loss_dict

class _DATrainer:
     def __init__(self, model, data_loader, optimizer, pseudo_labeler, backward_at_end=True):
          super().__init__(model, data_loader, optimizer, zero_grad_before_forward=not backward_at_end)
          self.pseudo_labeler = pseudo_labeler
          self.backward_at_end = backward_at_end

     def run_model(self, data):
          return run_model_labeled_unlabeled(self, *data)
     
     def do_backward(self, losses, override=False):
        """Disable the final backward pass if we are computing intermediate gradients in run_model.
        Can be overridden by setting override=True to always call superclass method."""
        if self.backward_at_end or override:
             super().do_backward(losses)

# Extend both Detectron2's AMPTrainer and SimpleTrainer classes with DA capabilities
# Used by DATrainer below in the same way DefaultTrainer uses the original AMP and Simple Trainers
class DAAMPTrainer(_DATrainer, AMPTrainer): pass
class DASimpleTrainer(_DATrainer, SimpleTrainer): pass

class DATrainer(DefaultTrainer):
     def _create_trainer(self, cfg, model, data_loader, optimizer):
          # build EMA model if applicable
          ema = EMA(build_model(cfg), cfg.EMA.ALPHA) if cfg.EMA.ENABLED else None
          # build pseudo-labeler if applicable (pseudo-labels creaed by: EMA, student, or None based on cfg)
          pseudo_labeler = PseudoLabeler(cfg, ema or model) # if cfg.DOMAIN_ADAPT.TEACHER.ENABLED else None
          trainer = (DAAMPTrainer if cfg.SOLVER.AMP.ENABLED else DASimpleTrainer)(model, data_loader, optimizer, pseudo_labeler,
                                                                                  backward_at_end=cfg.SOLVER.BACKWARD_AT_END)
          return trainer
     
     def _create_checkpointer(self, model, cfg):
          checkpointer = super(DATrainer, self)._create_checkpointer(model, cfg)
          if cfg.EMA.ENABLED:
               checkpointer.add_checkpointable("ema", self._trainer.pseudo_labeler.model)
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
                    self._last_eval_results = self.test(self.cfg, self._trainer.pseudo_labeler.model.model)
                    return self._last_eval_results
               eval_hook = hooks.EvalHook(self.cfg.TEST.EVAL_PERIOD, test_and_save_results_ema)
               if comm.is_main_process():
                    ret.insert(-1, eval_hook) # before PeriodicWriter if in main process
               else:
                    ret.append(eval_hook)

          # add a hook to save the best (teacher, if EMA enabled) checkpoint to model_best.pth
          if comm.is_main_process():
               for test_set in self.cfg.DATASETS.TEST:
                    ret.insert(-1, BestCheckpointer(self.cfg.TEST.EVAL_PERIOD, self.checkpointer,
                                                    f"{test_set}/bbox/AP50", "max", file_prefix=f"{test_set}_model_best"))

          return ret
     
     @classmethod
     def build_optimizer(cls, cfg, model):
          """
          - Change the learning rate to account for the fact that we are doing gradient accumulation in order
          to run multiple batches of labeled and unlabeled data each training step.
          - Enable use of alternative optimizers (e.g. AdamW for ViTDet)
          """
          logger = logging.getLogger("detectron2")
          effective_batch_size = sum([ int(r * cfg.SOLVER.IMS_PER_BATCH) for r in cfg.DATASETS.BATCH_RATIOS ])
          lr_scale = effective_batch_size / cfg.SOLVER.IMS_PER_BATCH
          logger.info(f"Effective batch size is {effective_batch_size}.")
          logger.info(f"Scaling LR from {cfg.SOLVER.BASE_LR} to {lr_scale * cfg.SOLVER.BASE_LR}.")

          cfg.defrost()
          cfg.SOLVER.BASE_LR = lr_scale * cfg.SOLVER.BASE_LR
          cfg.freeze()

          if cfg.SOLVER.OPTIMIZER.upper() == "SGD":
               return super(DATrainer, cls).build_optimizer(cfg, model)
          elif cfg.SOLVER.OPTIMIZER.upper() == "ADAMW":
               # TOOD: this could be cleaner and maybe removed
               if cfg.MODEL.BACKBONE.NAME == "build_vitdet_b_backbone":
                    return get_adamw_optim(model)
               elif cfg.MODEL.BACKBONE.NAME == "build_swinb_fpn_backbone":
                    return get_swinb_optim(model)
               else:
                    raise ValueError(f"Unknown backbone {cfg.MODEL.BACKBONE.NAME}.")
          else:
               raise ValueError(f"Unknown optimizer {cfg.SOLVER.OPTIMIZER}.")

     @classmethod
     def build_train_loader(cls, cfg):
          batch_contents = cfg.DATASETS.BATCH_CONTENTS
          batch_sizes = [ int(r * cfg.SOLVER.IMS_PER_BATCH) for r in cfg.DATASETS.BATCH_RATIOS ]
          assert len(batch_contents) == len(batch_sizes), "len(cfg.DATASETS.BATCH_CONTENTS) must equal len(cfg.DATASETS.BATCH_RATIOS)."
          labeled_bs = [batch_sizes[i] for i in range(len(batch_contents)) if batch_contents[i].startswith("labeled")]
          labeled_bs = max(labeled_bs) if len(labeled_bs) else 0
          unlabeled_bs = [batch_sizes[i] for i in range(len(batch_contents)) if batch_contents[i].startswith("unlabeled")]
          unlabeled_bs = max(unlabeled_bs) if len(unlabeled_bs) else 0

          # create labeled dataloader
          labeled_loader = None
          if labeled_bs > 0 and len(cfg.DATASETS.TRAIN):
               labeled_loader = build_detection_train_loader(get_detection_dataset_dicts(cfg.DATASETS.TRAIN, filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS), 
                    mapper=SaveWeakDatasetMapper(cfg, is_train=True, augmentations=get_augs(cfg, labeled=True, include_strong_augs="labeled_strong" in batch_contents),
                    dataset_type="source"),
                    num_workers=cfg.DATALOADER.NUM_WORKERS, 
                    total_batch_size=labeled_bs)

          # create unlabeled dataloader
          unlabeled_loader = None
          if unlabeled_bs > 0 and len(cfg.DATASETS.UNLABELED):
               unlabeled_loader = build_detection_train_loader(get_detection_dataset_dicts(cfg.DATASETS.UNLABELED, filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS), 
                    mapper=UnlabeledDatasetMapper(cfg, is_train=True, augmentations=get_augs(cfg, labeled=False, include_strong_augs="unlabeled_strong" in batch_contents),
                    dataset_type="target"),
                    num_workers=cfg.DATALOADER.NUM_WORKERS,
                    total_batch_size=unlabeled_bs)

          return WeakStrongDataloader(labeled_loader, unlabeled_loader, batch_contents)
     
     def before_step(self):
          super(DATrainer, self).before_step()
          if self.cfg.EMA.ENABLED:
               self._trainer.pseudo_labeler.model.update_weights(self._trainer.model, self.iter)

     def create_ddp_model(self, model, broadcast_buffers, cfg):
          """In progress: Add FSDP support.
          Not currently working.
          """
          if cfg.MODEL.FSDP_ENABLED and comm.get_world_size() > 1:
               return FSDP(model, 
                           auto_wrap_policy=always_wrap_policy
                           )
          else:
               return super(DATrainer, self).create_ddp_model(model, broadcast_buffers, cfg)
               