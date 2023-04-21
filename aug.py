import logging
import torchvision.transforms as transforms
    
# Modified from Adaptive Teacher / Unbiased Teacher codebase
# Changes:
#    - Remove conversion to PILImage
#    - Use Torchvision implementation of Gaussian blur with larger kernel_size/radius
def build_strong_augmentation():
    return transforms.Compose([
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(1,9), sigma=(0.1, 2.0))], p=0.5),
        transforms.RandomErasing(
                p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"
            ),
        transforms.RandomErasing(
                p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"
            ),
        transforms.RandomErasing(
                p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random"
            ),
    ])