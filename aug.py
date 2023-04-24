import torchvision.transforms as transforms
import torch
import warnings
    
from detectron2.data.transforms.augmentation_impl import RandomApply
from detectron2.data import transforms as T

from torch.nn import functional as F

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Modified from Adaptive Teacher / Unbiased Teacher codebase
# Changes:
#    - Remove conversion to PILImage
#    - Use Torchvision implementation of Gaussian blur with larger kernel_size/radius
#    - Disable hue adjustment in ColorJitter because it is very slow
def build_strong_augmentation():
    return transforms.Compose([
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)], p=0.8),
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

def mic_resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

def do_mic(img: torch.Tensor, ratio, block_size):
    """
    Do MIC augmentation on one image.
    """
    img = img.clone()
    _, H, W = img.shape

    mshape = 1, round(H / block_size), round(W / block_size)
    input_mask = torch.rand(mshape, device=img.device)
    input_mask = (input_mask > ratio).float().unsqueeze(0) # add unsqueeze to put batch dim back
    input_mask = mic_resize(input_mask, size=(H, W)).squeeze(0) # and remove batch dim before mul
    masked_img = img * input_mask

    return masked_img

# # TODO: Doesn't work because of numpy/PIL issue

# def build_strong_augmentation_detectron2():
#     return [
#         # weak; TODO from cfg settings
#         T.ResizeShortestEdge(short_edge_length=(800, 832, 864, 896, 928, 960, 992, 1024), max_size=2048, 
#                            sample_style='choice'), 
#         T.RandomFlip(),

#         # make compatible with torchvision
#         TorchvisionColorAugmentation(transforms.ToPILImage),

#         # # strong
#         # RandomColorJitter(prob=0.8, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
#         # RandomGrayscale(prob=0.2),
#         # RandomGaussianBlur(prob=0.5),
#         # RandomErasing(prob=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"),
#         # RandomErasing(prob=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"),
#         # RandomErasing(prob=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random"),

#         # and back
#         # TorchvisionColorAugmentation(transforms.ToTensor), # -> C x H x W
#         # TorchvisionColorAugmentation(partial(torch.permute, dims=[1,2,0]))#torchvision.ops.Permute([1,2,0])) # -> H x W x C
#         ToTensorPermute()
#     ]


# # Wrapper for torchvision color transforms (i.e. transforms that only modify RGB values)
# # to function with Detectron2's Augmentation API
# class TorchvisionColorAugmentation(T.Augmentation):
#     def __init__(self, transform, **transform_kwargs):
#         super().__init__()
#         print(transform)
#         print(transform_kwargs)
#         self.transform = transform(**transform_kwargs)

#     def get_transform(self, image):
#         return T.ColorTransform(self.transform)

# class RandomTorchvisionColorAugmentation(TorchvisionColorAugmentation):
#     def __init__(self, transform, prob=0.5, **transform_kwargs):
#         super().__init__(transform, **transform_kwargs)
#         self.prob = prob

#     # see if the RandomApply is causing the problem
#     # def get_transform(self, image):
#         # return T.ColorTransform(RandomApply(super().get_transform(image), prob=self.prob))

# class RandomColorJitter(RandomTorchvisionColorAugmentation):
#     def __init__(self, prob=0.5, brightness=0, contrast=0, saturation=0, hue=0):
#         super().__init__(transforms.ColorJitter, prob=prob, brightness=brightness, contrast=contrast, 
#                          saturation=saturation, hue=hue)
        
# class RandomGrayscale(RandomTorchvisionColorAugmentation):
#     def __init__(self, prob=0.5):
#         super().__init__(transforms.RandomGrayscale, prob=prob)

# class RandomGaussianBlur(RandomTorchvisionColorAugmentation):
#     def __init__(self, kernel_size=3, sigma=(0.1, 2.0), prob=0.5):
#         super().__init__(transforms.GaussianBlur, prob=prob, kernel_size=kernel_size, sigma=sigma)

# class RandomErasing(RandomTorchvisionColorAugmentation):
#     def __init__(self, prob=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value="random"):
#         super().__init__(transforms.RandomErasing, prob=prob, scale=scale, ratio=ratio, value=value)

# class ToTensorPermute(TorchvisionColorAugmentation):
#     def __init__(self):
#         super().__init__(transforms.ToTensor)

#     def get_transform(self, image):
#         return T.TransformList([
#             T.ColorTransform(transforms.PILToTensor()),
#             T.ColorTransform(partial(torch.permute, dims=[1,2,0]))
#         ]) 