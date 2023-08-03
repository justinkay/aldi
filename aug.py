import numpy as np
import random
import math
from scipy.ndimage import gaussian_filter
import cv2

from detectron2.data.transforms.augmentation import _get_aug_input_args
from detectron2.data.transforms.augmentation_impl import RandomApply
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from fvcore.transforms.transform import Transform, NoOpTransform

# key for weakly augmented image in aug_input object
WEAK_IMG_KEY = "img_weak"

def get_augs(cfg, labeled, include_strong_augs=True):
    """
    Get augmentations list for a dataset (labeled or unlabeled) according to settings in cfg.
    """
    # default weak augmentations: see DatasetMapper.from_config
    augs = utils.build_augmentation(cfg, is_train=True)
    if cfg.INPUT.CROP.ENABLED:
        augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
    
    # add a hook to save the image after weak augmentation occurs
    augs.append(SaveImgAug(WEAK_IMG_KEY))

    # add strong augmentation
    if include_strong_augs:
        augs += build_strong_augmentation(include_erasing=cfg.AUG.INCLUDE_RANDOM_ERASING)

        # add MIC
        if (labeled and cfg.AUG.LABELED_MIC_AUG) or (not labeled and cfg.AUG.UNLABELED_MIC_AUG):
            augs.append(T.RandomApply(MICTransform(cfg.AUG.MIC_RATIO, cfg.AUG.MIC_BLOCK_SIZE), prob=1.0))

    return augs

def build_strong_augmentation(include_erasing=True):
    """
    Modified from Adaptive Teacher / Unbiased Teacher codebase
        - Remove random hue transform (it was very slow)
        - Use scipy implementation of gaussian blur to avoid converting from PIL to numpy and back
    """
    augs = [
        T.RandomApply(T.AugmentationList([
            T.RandomContrast(0.6, 1.4),
            T.RandomBrightness(0.6, 1.4),
            T.RandomSaturation(0.6, 1.4),
        ]), prob=0.8),
        T.RandomApply(T.RandomSaturation(0, 0), prob=0.2), # Random grayscale
        T.RandomApply(RandomBlurTransform((0.1, 2.0)), prob=0.5),
    ]
    if include_erasing:
        re0 = RandomEraseTransform(sl=0.05, sh=0.2, r1=0.3, r2=3.3, value="random")
        re1 = RandomEraseTransform(sl=0.02, sh=0.2, r1=0.1, r2=6, value="random")
        re2 = RandomEraseTransform(sl=0.02, sh=0.2, r1=0.05, r2=8, value="random")
        re_augs = [
            RandomApplyPrecompute(re0, prob=0.7),
            RandomApplyPrecompute(re1, prob=0.5),
            RandomApplyPrecompute(re2, prob=0.3),
        ]

        # store these so we can use them for processing pseudo labels later
        xtra = []
        for i, aug in enumerate(re_augs):
            if aug.precomputed < aug.prob:
                xtra.append(SaveErasureAug(f"erased_{i}", aug.aug.tfm))
        augs += re_augs + xtra
        
    return augs

class RandomApplyPrecompute(T.RandomApply):
    def __init__(self, tfm_or_aug, prob=0.5):
        super().__init__(tfm_or_aug, prob)
        self.precomputed = self._rand_range(do=True)

    def _rand_range(self, low=1.0, high=None, size=None, do=False):
        if do:
            return super()._rand_range(low, high, size)
        else:
            return self.precomputed

class SaveImgAug(T.Augmentation):
    """
    A Detectron2 'augmentation' that saves a copy of the image to the input object.
    This is used to get a copy of the image before additional augmentations are applied,
    so that, e.g., we can obtain a weakly and strongly augmented version of the same image for 
    self-training.
    """
    def __init__(self, savename):
        super().__init__()
        self._init(locals())

    def get_transform(self, *args) -> Transform:
        return NoOpTransform()

    def __call__(self, aug_input) -> Transform:
        image = _get_aug_input_args(self, aug_input)[0].copy()
        setattr(aug_input, self.savename, image)
        return super().__call__(aug_input)

class SaveErasureAug(T.Augmentation):
    def __init__(self, savename, tfm):
        super().__init__()
        self._init(locals())

    def get_transform(self, *args) -> Transform:
        return NoOpTransform()

    def __call__(self, aug_input) -> Transform:
        image = _get_aug_input_args(self, aug_input)[0].copy()
        imgh, imgw, _ = image.shape
        setattr(aug_input, self.savename, self.tfm.get_erased_box_coords(imgh, imgw))
        return super().__call__(aug_input)

class RandomBlurTransform(Transform):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        if img.dtype == np.uint8:
            img = img.astype(np.float32)
            img = gaussian_filter(img, sigma=sigma)
            return np.clip(img, 0, 255).astype(np.uint8)
        else:
            return gaussian_filter(img, sigma=sigma)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation

    def inverse(self) -> Transform:
        return NoOpTransform()

class RandomEraseTransform(Transform):
    """
    Modified from this implementation: https://github.com/zhunzhong07/Random-Erasing/blob/master/transforms.py
    to better match Torchvision params:
        scale=(sl, sh): range of proportion of erased area against input image.
        ratio=(r1. r2): range of aspect ratio of erased area.
        value="random" or specified value [0,1)
    """
    def __init__(self, sl = 0.02, sh = 0.4, r1 = 0.3, r2=3.3, value="random", anno_aware=True): #False): #[0.4914, 0.4822, 0.4465]):
        super().__init__()
        self._set_attributes(locals())

        # initialize these ahead of time
        self.area = random.uniform(self.sl, self.sh)
        self.aspect_ratio = random.uniform(self.r1, self.r2)
        self.h0 = random.random()
        self.w0 = random.random()
    
    def apply_image(self, img: np.ndarray) -> np.ndarray:
        was_int = False
        if img.dtype == np.uint8:
            was_int = True
            img = img.astype(np.float32)

        imgh, imgw, c  = img.shape
        self.last_applied = (x0, y0, x1, y1) = self.get_erased_box_coords(imgh, imgw)

        if self.value == "random":
            img[y0:y1, x0:x1, :] = np.random.rand(y1-y0, x1-x0, c)
        else:
            img[y0:y1, x0:x1, :] = self.value

        if was_int:
            img[y0:y1, x0:x1, :] *= 255 # put mask values in range [0,255]
            return np.clip(img, 0, 255).astype(np.uint8)
        else:
            return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation

    def inverse(self) -> Transform:
        return NoOpTransform() # ?

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        trans_boxes = super().apply_box(box)
        if self.anno_aware:
            # inefficient way of doing this -- change to batched?
            # could also make a threshold -- remove boxes more than 95% erased
            for i, box in enumerate(trans_boxes):
                trans_boxes[i] = self.modify_erased_annotation(box, self.last_applied)
        return trans_boxes

    def get_erased_box_coords(self, imgh, imgw):
        area = imgw * imgh
        target_area = self.area * area 
        aspect_ratio = self.aspect_ratio
        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))
        h = max(1, min(h, imgh-1))
        w = max(1, min(w, imgw-1))
        h0 = int(self.h0 * (imgh - h - 1))
        w0 = int(self.w0 * (imgw - w - 1))
        return np.asarray([w0, h0, w0+w, h0+h])

    @classmethod
    def modify_erased_annotation(cls, anno, erased_box_coords):
        x0, y0, x1, y1 = anno

        # if completely enclosed, delete it
        if x0 >= erased_box_coords[0] and y0 >= erased_box_coords[1] and x1 <= erased_box_coords[2] and y1 <= erased_box_coords[3]:
            return np.asarray([-1,-1,-1,-1])
        
        # if left side is inside erased_box, move it to be flush with the right side of erased_box
        if x0 >= erased_box_coords[0] and x0 <= erased_box_coords[2] and y0 >= erased_box_coords[1] and y1 <= erased_box_coords[3]:
            x0 = erased_box_coords[2]

        # if right side is inside erased_box, move it to be flush with the left side of erased_box
        if x1 >= erased_box_coords[0] and x1 <= erased_box_coords[2] and y0 >= erased_box_coords[1] and y1 <= erased_box_coords[3]:
            x1 = erased_box_coords[0]

        # if top side is inside erased_box, move it to be flush with the bottom side of erased_box
        if x0 >= erased_box_coords[0] and x1 <= erased_box_coords[2] and y0 >= erased_box_coords[1] and y0 <= erased_box_coords[3]:
            y0 = erased_box_coords[3]

        # if bottom side is inside erased_box, move it to be flush with the top side of erased_box
        if x0 >= erased_box_coords[0] and x1 <= erased_box_coords[2] and y1 >= erased_box_coords[1] and y1 <= erased_box_coords[3]:
            y1 = erased_box_coords[1]

        return np.asarray([x0, y0, x1, y1])

class MICTransform(Transform):
    def __init__(self, ratio, block_size):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        H, W, C = img.shape

        was_int = False
        if img.dtype == np.uint8:
            was_int = True
            img = img.astype(np.float32)

        mh, mw = round(H / self.block_size), round(W / self.block_size)
        input_mask = np.random.rand(mh, mw)
        input_mask = input_mask > self.ratio
        input_mask = cv2.resize(np.asarray(input_mask, dtype="uint8"), (W,H), interpolation=cv2.INTER_NEAREST)
        masked_img = img * np.repeat(input_mask[..., np.newaxis], C, axis=-1)

        if was_int:
            return np.clip(masked_img, 0, 255).astype(np.uint8)
        else:
            return masked_img  

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation

    def inverse(self) -> Transform:
        return NoOpTransform()