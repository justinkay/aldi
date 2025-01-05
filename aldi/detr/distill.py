from aldi.distill import DISTILL_MIXIN_REGISTRY

from .libs.DeformableDETRDetectron2.meta_arch import DeformableDETR


@DISTILL_MIXIN_REGISTRY.register()
class DETRDistillMixin(DeformableDETR): pass