import torch
from typing import Any, Dict

from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.checkpoint.c2_model_loading import align_and_update_state_dicts


class DetectionCheckpointerWithEMA(DetectionCheckpointer):
    """The default DetectionCheckpointer will load from the 'model' entry in
    the checkpoint file. This is not desirable if you want to initialize from a model
    that was trained (i.e. burned-in) with EMA. This class will load the EMA model instead
    at the beginning of training.
    This behavior can be disabled by setting cfg..EMA.LOAD_FROM_EMA_ON_START = False.
    """
    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        super().__init__(model, save_dir=save_dir, save_to_disk=save_to_disk, **checkpointables)

    def resume_or_load(self, path: str, *, resume: bool = True) -> Dict[str, Any]:
        if resume or (not path.endswith(".pth")) or "ema" not in torch.load(path, map_location=torch.device("cpu")).keys():
            return super().resume_or_load(path, resume=resume)
        else:
            self.load(path, checkpointables=["ema"])
            self.logger.info("Loading EMA weights as model starting point.")
            self.model.load_state_dict(self.checkpointables["ema"].model.state_dict())