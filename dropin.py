##############
# This file contains drop-in replacements for components in Detectron2.
# Comoponents in this file **do not change the original functionality** of Detectron2,
# but often add hooks into the original functionality to reduce the need to copy-paste 
# code when subclassing.
##############

import logging
import weakref
import time
import torch
import copy

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.dataset_mapper import DatasetMapper as _DatasetMapper
from detectron2.data import detection_utils as utils
from detectron2.engine.train_loop import TrainerBase
from detectron2.engine.train_loop import AMPTrainer as _AMPTrainer
from detectron2.engine.train_loop import SimpleTrainer as _SimpleTrainer
from detectron2.engine.defaults import create_ddp_model
from detectron2.engine.defaults import DefaultTrainer as _DefaultTrainer
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm
from detectron2.utils.events import get_event_storage

class DefaultTrainer(_DefaultTrainer):
    """
    Same as detectron2.engine.defaults.DefaultTrainer, but adds a _create_trainer method
    to allow easier use of other trainers.
    """

    def __init__(self, cfg):
            """
            Args:
                cfg (CfgNode):
            """
            # call grandparent init, so we can overwrite the DefaultTrainer __init__ logic completely
            TrainerBase.__init__(self) 
            
            logger = logging.getLogger("detectron2")
            if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
                setup_logger()
            cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

            # Assume these objects must be constructed in this order.
            model = self.build_model(cfg)
            optimizer = self.build_optimizer(cfg, model)
            data_loader = self.build_train_loader(cfg)

            model = create_ddp_model(model, broadcast_buffers=False)

            ## Change is here ##
            self._trainer = self._create_trainer(cfg, model, data_loader, optimizer)
            ##   End change   ##

            self.scheduler = self.build_lr_scheduler(cfg, optimizer)
            self.checkpointer = DetectionCheckpointer(
                # Assume you want to save checkpoints together with logs/statistics
                model,
                cfg.OUTPUT_DIR,
                trainer=weakref.proxy(self),
            )
            self.start_iter = 0
            self.max_iter = cfg.SOLVER.MAX_ITER
            self.cfg = cfg

            self.register_hooks(self.build_hooks())

    def _create_trainer(self, cfg, model, data_loader, optimizer):
         return (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
                model, data_loader, optimizer
            )
    
class SimpleTrainer(_SimpleTrainer):
    """
    Same as detectron2.engine.train_loop.SimpleTrainer, but adds a run_model method
    that provides a way to change the model's forward pass without having to copy-paste
    the entire run_step method.
    """
    def run_step(self):
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        if self.zero_grad_before_forward:
            self.optimizer.zero_grad()

        ## Change is here ##
        loss_dict = self.run_model(data)
        ##   End change   ##

        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())
        if not self.zero_grad_before_forward:
            """
            If you need to accumulate gradients or do something similar, you can
            wrap the optimizer with your custom `zero_grad()` method.
            """
            self.optimizer.zero_grad()
        losses.backward()

        self.after_backward()

        self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()
    
    def run_model(self, data):
        return self.model(data)


class AMPTrainer(_AMPTrainer):
    """
    Same as detectron2.engine.train_loop.AMPTrainer, but adds a run_model method
    that provides a way to change the model's forward pass without having to copy-paste
    the entire run_step method.
    """
    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        if self.zero_grad_before_forward:
            self.optimizer.zero_grad()
        with autocast(dtype=self.precision):
            
            ## Change is here ##
            loss_dict = self.run_model(data)
            ##   End change   ##

            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        if not self.zero_grad_before_forward:
            self.optimizer.zero_grad()

        self.grad_scaler.scale(losses).backward()

        if self.log_grad_scaler:
            storage = get_event_storage()
            storage.put_scalar("[metric]grad_scaler", self.grad_scaler.get_scale())

        self.after_backward()

        self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

    def run_model(self, data):
        return self.model(data)
    
    
class DatasetMapper(_DatasetMapper):
    def __call__(self, dataset_dict):
        """
        Same as detectron2.data.dataset_mapper.DatasetMapper, but adds a way to
        access the aug_input object in subclasses without copy-pasting the entire
        __call__ method.
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        ## Change is here ##
        dataset_dict = self._after_call(dataset_dict, aug_input)
        ##   End change   ##

        return dataset_dict
    
    def _after_call(self, dataset_dict, aug_input):
        return dataset_dict