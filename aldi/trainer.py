import os
import copy
from torch.nn.parallel import DistributedDataParallel as DDP

from detectron2.data.build import build_detection_train_loader, get_detection_dataset_dicts
from detectron2.engine import hooks, BestCheckpointer
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from detectron2.modeling.meta_arch.build import build_model
from detectron2.solver import build_optimizer
from detectron2.utils.events import get_event_storage
from detectron2.utils import comm

from aldi.aug import WEAK_IMG_KEY, get_augs
from aldi.backbone import get_adamw_optim
from aldi.distill import Distiller
from aldi.dropin import DefaultTrainer, AMPTrainer, SimpleTrainer
from aldi.dataloader import SaveWeakDatasetMapper, UMTDatasetMapper, UnlabeledSaveWeakDatasetMapper, UnlabeledUMTDatasetMapper, WeakStrongDataloader
from aldi.ema import EMA


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


DEBUG = False
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
          model_batch_size (int): batch size to feed to the model *per GPU*
     """
     model = trainer.model
     backward_at_end = trainer.backward_at_end
     model_batch_size = trainer.model_batch_size # TODO this could be None

     _model = model.module if type(model) == DDP else model
     do_weak = labeled_weak is not None
     do_strong = labeled_strong is not None
     do_align = any( [ getattr(_model, a, None) is not None for a in ["img_align", "ins_align"] ] )
     do_distill = trainer.distiller.distill_enabled()

     total_batch_size = sum([len(s or []) for s in [labeled_weak, labeled_strong, unlabeled_weak]])
     num_grad_accum_steps = total_batch_size // model_batch_size

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
                    if not backward_at_end: 
                         v = v.detach()
                    loss_dict[f"{k}_{suffix}"] = loss_dict.get(f"{k}_{suffix}", 0) + v

     def maybe_do_backward(losses, key_conditional=lambda k: True):
          """Helper method to do backward pass if not doing it at the end."""
          if not backward_at_end:
               losses = { k: v * 0 if not key_conditional(k) else v for k, v in losses.items() }
               trainer.do_backward(sum(losses.values()) / num_grad_accum_steps, override=True)

     def do_training_step(data, name="", key_conditional=lambda k: True, **kwargs):
          """Helper method to do a forward pass:
               - Handle gradient accumulation and possible backward passes
               - Handle Detectron2's loss dictionary
          """
          for batch_i in range(0, len(data), model_batch_size):
               loss = model(data[batch_i:batch_i+model_batch_size], **kwargs)
               maybe_do_backward(loss, key_conditional)
               add_to_loss_dict(loss, name, key_conditional)

     def do_distill_step(teacher_data, student_data, name="", key_conditional=lambda k: True, **kwargs):
          assert len(teacher_data) == len(student_data), "Teacher and student data must be the same length."
          for batch_i in range(0, len(teacher_data), model_batch_size):
               distill_loss = trainer.distiller(teacher_data[batch_i:batch_i+model_batch_size], 
                                                student_data[batch_i:batch_i+model_batch_size])
               maybe_do_backward(distill_loss, key_conditional)
               add_to_loss_dict(distill_loss, name, key_conditional)

     # Weakly-augmented source imagery (Used for normal training and/or domain alignment)
     if do_weak: 
          do_training_step(labeled_weak, "source_weak", lambda k: do_weak or (do_align and "_da_" in k), do_align=do_align)
     
     # Strongly-augmented source imagery (Used for normal training and/or domain alignment)
     if do_strong:
          do_training_step(labeled_strong, "source_strong", lambda k: do_strong or (do_align and "_da_" in k), do_align=do_align)

     # Weakly-augmented target imagery (Only used for domain alignment)
     if do_align: 
          do_training_step(unlabeled_weak, "target_weak", lambda k: "_da_" in k, labeled=False, do_align=True)

     # Distillation losses
     if do_distill:
          do_distill_step(unlabeled_weak, unlabeled_strong, "distill")
          # if DEBUG: 
          #   debug_dict['last_pseudolabeled'] = copy.deepcopy(pseudolabeled_data)

     return loss_dict


# Extend both Detectron2's AMPTrainer and SimpleTrainer classes with DA capabilities
# Used by DATrainer below in the same way DefaultTrainer uses the original AMP and Simple Trainers
class _DATrainer:
     def __init__(self, model, data_loader, optimizer, distiller, backward_at_end=True, model_batch_size=None):
          super().__init__(model, data_loader, optimizer, zero_grad_before_forward=not backward_at_end)
          self.distiller = distiller
          self.backward_at_end = backward_at_end
          self.model_batch_size = model_batch_size

     def run_model(self, data):
          return run_model_labeled_unlabeled(self, *data)
     
     def do_backward(self, losses, override=False):
        """Disable the final backward pass if we are computing intermediate gradients in run_model.
        Can be overridden by setting override=True to always call superclass method."""
        if self.backward_at_end or override:
             super().do_backward(losses)
class DAAMPTrainer(_DATrainer, AMPTrainer): pass
class DASimpleTrainer(_DATrainer, SimpleTrainer): pass

class DATrainer(DefaultTrainer):
     """Modified DefaultTrainer to support Mean Teacher style training."""
     def _create_trainer(self, cfg, model, data_loader, optimizer):
          # build EMA model if applicable
          self.ema = EMA(build_model(cfg), cfg.EMA.ALPHA) if cfg.EMA.ENABLED else None
          distiller = Distiller.from_config(teacher=self.ema.model if cfg.EMA.ENABLED else model, student=model, cfg=cfg)
          trainer = (DAAMPTrainer if cfg.SOLVER.AMP.ENABLED else DASimpleTrainer)(model, data_loader, optimizer, distiller,
                                                                                  backward_at_end=cfg.SOLVER.BACKWARD_AT_END,
                                                                                  model_batch_size=cfg.SOLVER.IMS_PER_GPU)
          return trainer
     
     def _create_checkpointer(self, model, cfg):
          checkpointer = super(DATrainer, self)._create_checkpointer(model, cfg)
          if cfg.EMA.ENABLED:
               checkpointer.add_checkpointable("ema", self.ema)
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
                    self._last_eval_results = self.test(self.cfg, self.ema.model)
                    return self._last_eval_results
               eval_hook = hooks.EvalHook(self.cfg.TEST.EVAL_PERIOD, test_and_save_results_ema)
               if comm.is_main_process():
                    ret.insert(-1, eval_hook) # before PeriodicWriter if in main process
               else:
                    ret.append(eval_hook)

          # add a hook to save the best (teacher, if EMA enabled) checkpoint to model_best.pth
          if comm.is_main_process():
               if len(self.cfg.DATASETS.TEST) == 1:
                    ret.insert(-1, BestCheckpointer(self.cfg.TEST.EVAL_PERIOD, self.checkpointer,
                                                    f"bbox/AP50", "max", file_prefix=f"{self.cfg.DATASETS.TEST[0]}_model_best"))
               else: 
                    for test_set in self.cfg.DATASETS.TEST:
                         ret.insert(-1, BestCheckpointer(self.cfg.TEST.EVAL_PERIOD, self.checkpointer,
                                                    f"{test_set}/bbox/AP50", "max", file_prefix=f"{test_set}_model_best"))

          return ret
     
     @classmethod
     def build_optimizer(cls, cfg, model):
          """
          - Enable use of alternative optimizers (e.g. AdamW for ViTDet)
          """
          if cfg.SOLVER.OPTIMIZER.upper() == "SGD":
               return super(DATrainer, cls).build_optimizer(cfg, model)
          elif cfg.SOLVER.OPTIMIZER.upper() == "ADAMW":
               # TOOD: this could be cleaner and maybe removed
               if cfg.MODEL.BACKBONE.NAME == "build_vitdet_b_backbone":
                    return get_adamw_optim(model, include_vit_lr_decay=True)
               elif cfg.MODEL.BACKBONE.NAME == "build_swinb_fpn_backbone":
                    return get_swinb_optim(model)
               else:
                    raise ValueError(f"Unknown backbone {cfg.MODEL.BACKBONE.NAME}.")
          else:
               raise ValueError(f"Unknown optimizer {cfg.SOLVER.OPTIMIZER}.")

     @classmethod
     def build_train_loader(cls, cfg):
          batch_contents = cfg.DATASETS.BATCH_CONTENTS
          batch_ratios = cfg.DATASETS.BATCH_RATIOS
          total_batch_size = cfg.SOLVER.IMS_PER_BATCH
          batch_sizes = [ int(total_batch_size * r / sum(batch_ratios)) for r in batch_ratios ]
          assert len(batch_contents) == len(batch_sizes), "len(cfg.DATASETS.BATCH_CONTENTS) must equal len(cfg.DATASETS.BATCH_RATIOS)."
          assert sum(batch_sizes) == total_batch_size, f"sum(batch_sizes)={sum(batch_sizes)} must equal total_batch_size={total_batch_size}"

          labeled_bs = [batch_sizes[i] for i in range(len(batch_contents)) if batch_contents[i].startswith("labeled")]
          labeled_bs = max(labeled_bs) if len(labeled_bs) else 0
          unlabeled_bs = [batch_sizes[i] for i in range(len(batch_contents)) if batch_contents[i].startswith("unlabeled")]
          unlabeled_bs = max(unlabeled_bs) if len(unlabeled_bs) else 0

          # create labeled dataloader
          labeled_loader = None
          if labeled_bs > 0 and len(cfg.DATASETS.TRAIN):
               DatasetMapper = SaveWeakDatasetMapper if not cfg.MODEL.UMT.ENABLED else UMTDatasetMapper
               labeled_loader = build_detection_train_loader(get_detection_dataset_dicts(cfg.DATASETS.TRAIN, filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS), 
                    mapper=DatasetMapper(cfg, is_train=True, augmentations=get_augs(cfg, labeled=True, include_strong_augs="labeled_strong" in batch_contents and not cfg.MODEL.UMT.ENABLED),
                    dataset_type="source"),
                    num_workers=cfg.DATALOADER.NUM_WORKERS, 
                    total_batch_size=labeled_bs)

          # create unlabeled dataloader
          unlabeled_loader = None
          if unlabeled_bs > 0 and len(cfg.DATASETS.UNLABELED):
               UnlabeledDatasetMapper = UnlabeledSaveWeakDatasetMapper if not cfg.MODEL.UMT.ENABLED else UnlabeledUMTDatasetMapper
               unlabeled_loader = build_detection_train_loader(get_detection_dataset_dicts(cfg.DATASETS.UNLABELED, filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS), 
                    mapper=UnlabeledDatasetMapper(cfg, is_train=True, augmentations=get_augs(cfg, labeled=False, include_strong_augs="unlabeled_strong" in batch_contents),
                    dataset_type="target"),
                    num_workers=cfg.DATALOADER.NUM_WORKERS,
                    total_batch_size=unlabeled_bs)

          return WeakStrongDataloader(labeled_loader, unlabeled_loader, batch_contents)
     
     def before_step(self):
          """Update the EMA model every step."""
          super(DATrainer, self).before_step()
          if self.cfg.EMA.ENABLED:
               self.ema.update_weights(self._trainer.model, self.iter)
               