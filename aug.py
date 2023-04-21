import logging
import torchvision.transforms as transforms
import torch
import random
from PIL import ImageFilter

# From Adaptive Teacher codebase
class GaussianBlur:
    """
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    Adapted from MoCo:
    https://github.com/facebookresearch/moco/blob/master/moco/loader.py
    Note that this implementation does not seem to be exactly the same as
    described in SimCLR.
    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
# From Adaptive Teacher codebase
def build_strong_augmentation():
    """
    Create a list of :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """

    logger = logging.getLogger(__name__)
    augmentation = []
    # This is similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation.append(
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
    )
    augmentation.append(transforms.RandomGrayscale(p=0.2))
    augmentation.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))

    randcrop_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomErasing(
                p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"
            ),
            transforms.RandomErasing(
                p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"
            ),
            transforms.RandomErasing(
                p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random"
            ),
            transforms.ToPILImage(),
        ]
    )
    augmentation.append(randcrop_transform)
    return transforms.Compose(augmentation)

def apply_aug_to_batch(batch, aug):
    """Apply augmentation to batch of images."""
    imgs = torch.cat([i['image'] for i in batch])
    imgs = aug(imgs)
    for i, img in enumerate(imgs):
        batch[i]['image'] = img
    return batch