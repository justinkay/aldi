import os
import torch

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from detectron2.modeling.meta_arch.build import build_model

from dataloader import DatasetMapperTwoCropSeparate, build_detection_semisup_train_loader_two_crops
from mean_teacher import EMATeacher

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
        Student is already burned in by another trainer (?).
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
        if cfg.DOMAIN_ADAPT.EMA.ENABLED:
            self.ema = EMATeacher(build_model(cfg), cfg.DOMAIN_ADAPT.EMA.ALPHA)

    # TODO: JUST TAKEN FROM ADAPTIVE TEACHER FOR NOW
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparate(cfg, True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    def before_step(self):
        """
        - Update the teacher
        - Prefetch dataloader batch and add pseudo labels from teacher
        """
        super().before_step()

        # EMA update
        if self.cfg.DOMAIN_ADAPT.EMA.ENABLED:
            self.ema.update_weights(self.model, self.iter)

        # Teacher-student self-training
        if self.cfg.DOMAIN_ADAPT.TEACHER.ENABLED:
            # Prefetch dataloader batch and add pseudo labels from teacher
            label_strong, label_weak, unlabeled_strong, unlabeled_weak = self.data_loader.prefetch_batch()
            with torch.no_grad():
                # run teacher on weakly augmented data
                pseudo_labels = self.ema(unlabeled_weak)

                # add pseudo labels as ground truth for strongly augmented data
                for d, l in zip(unlabeled_strong, pseudo_labels):
                    print("D BEFORE", d)
                    d["instances"] = l["instances"]
                    print("D AFTER", d)
