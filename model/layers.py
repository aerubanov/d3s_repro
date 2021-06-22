import torch.nn as nn
import torch


def conv(in_channels: int, out_channels: int, kernel_size: int = 3,
        stride: int = 1, padding: int = 1, dilation: int = 1) -> nn.Sequential:
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                kernel_size, stride, padding, dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            )


def conv_no_relu(in_channels: int, out_channels: int, kernel_size: int = 3,
        stride: int = 1, padding: int = 1, dilation: int = 1) -> nn.Sequential:
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                kernel_size, stride, padding, dilation),
            nn.BatchNorm2d(out_channels),
            )


def valid_roi(roi: torch.Tensor, image_size: torch.Tensor) -> bool:
    valid = all(0 <= roi[:, 1]) and all(0 <= roi[:, 2]) and all(roi[:, 3] <= image_size[0]-1) and \
            all(roi[:, 4] <= image_size[1]-1)
    return valid


def normalize_vis_img(x):
    x = x - np.min(x)
    x = x / np.max(x)
    return (x * 255).astype(np.uint8)
