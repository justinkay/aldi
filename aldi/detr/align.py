from aldi.align import ALIGN_MIXIN_REGISTRY

from .libs.DeformableDETRDetectron2.modeling.meta_arch.deformable_detr import DeformableDETR

@ALIGN_MIXIN_REGISTRY.register()
class DETRAlignMixin(DeformableDETR): pass