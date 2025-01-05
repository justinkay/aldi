from aldi.align import ALIGN_MIXIN_REGISTRY

from .libs.DeformableDETRDetectron2.meta_arch import DeformableDETR


@ALIGN_MIXIN_REGISTRY.register()
class DETRAlignMixin(DeformableDETR): pass