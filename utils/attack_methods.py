import torch
import numpy as np
import scipy.stats as st
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms

from torchvision.transforms import functional

"""Translation-Invariant https://arxiv.org/abs/1904.02884"""
def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel.astype(np.float32)
    gaussian_kernel = np.stack([kernel, kernel, kernel])  # 5*5*3
    gaussian_kernel = np.expand_dims(gaussian_kernel, 1)  # 1*5*5*3
    gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()  # tensor and cuda
    return gaussian_kernel

"""Input diversity: https://arxiv.org/abs/1803.06978"""
def DI(x, resize_rate=1.15, diversity_prob=0.5):
    assert resize_rate >= 1.0
    assert diversity_prob >= 0.0 and diversity_prob <= 1.0
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
    ret = padded if torch.rand(1) < diversity_prob else x
    ret = transforms.Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST)(ret)
    return ret


class CropShift(torch.nn.Module):
    def __init__(self, low, high=None):
        super().__init__()
        high = low if high is None else high
        self.low, self.high = int(low), int(high)

    def sample_top(self, x, y):
        x = torch.randint(0, x + 1, (1,)).item()
        y = torch.randint(0, y + 1, (1,)).item()
        return x, y

    def forward(self, img):
        if self.low == self.high:
            strength = self.low
        else:
            strength = torch.randint(self.low, self.high, (1,)).item()

        w, h = functional.get_image_size(img)
        crop_x = torch.randint(0, strength + 1, (1,)).item()
        crop_y = strength - crop_x
        crop_w, crop_h = w - crop_x, h - crop_y

        top_x, top_y = self.sample_top(crop_x, crop_y)

        img = functional.crop(img, top_y, top_x, crop_h, crop_w)
        img = functional.pad(img, padding=[crop_x, crop_y], fill=0)

        top_x, top_y = self.sample_top(crop_x, crop_y)

        return functional.crop(img, top_y, top_x, h, w)
