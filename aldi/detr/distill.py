from aldi.distill import DISTILL_MIXIN_REGISTRY

from .libs.DeformableDETRDetectron2.modeling.meta_arch.deformable_detr import DeformableDETR


@DISTILL_MIXIN_REGISTRY.register()
class DETRDistillMixin(DeformableDETR): pass