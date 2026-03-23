"""ResNet-9 block generator for CycleGAN 256x256 image translation.

Implements the standard CycleGAN generator architecture with:
- Reflection-padded encoder with two downsampling stages
- Nine residual blocks for feature transformation
- Two upsampling stages with transposed convolutions
- Tanh output activation for [-1, 1] range
"""

from functools import partial
from typing import Callable, Type

import torch
import torch.nn as nn
from torch.nn import init

from src.config import ModelConfig


def get_norm_layer(norm_type: str) -> Callable[..., nn.Module]:
    """Return a normalization layer factory based on the specified type.

    Args:
        norm_type: Type of normalization. One of 'instance' or 'batch'.

    Returns:
        A partial function that creates the requested normalization layer.

    Raises:
        NotImplementedError: If norm_type is not recognized.
    """
    if norm_type == "instance":
        return partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == "batch":
        return partial(nn.BatchNorm2d)
    else:
        raise NotImplementedError(f"Normalization layer '{norm_type}' is not supported.")


def init_weights(net: nn.Module, init_type: str = "normal", init_gain: float = 0.02) -> None:
    """Initialize network weights using the specified strategy.

    Args:
        net: The network whose weights will be initialized.
        init_type: Initialization method. One of 'normal', 'xavier', 'kaiming'.
        init_gain: Scaling factor for normal and xavier initialization.

    Raises:
        NotImplementedError: If init_type is not recognized.
    """

    def _init_func(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            else:
                raise NotImplementedError(
                    f"Initialization method '{init_type}' is not supported."
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(_init_func)


class ResnetBlock(nn.Module):
    """Residual block with reflection padding, convolution, and instance norm.

    Architecture: ReflectionPad -> Conv -> Norm -> ReLU -> ReflectionPad -> Conv -> Norm + skip.
    """

    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module]) -> None:
        """Initialize the residual block.

        Args:
            dim: Number of channels in the convolutional layers.
            norm_layer: Factory function for normalization layers.
        """
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            norm_layer(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            norm_layer(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of same shape with residual added.
        """
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    """ResNet-based generator for CycleGAN with 9 residual blocks.

    Architecture:
        Encoder: ReflectionPad(3) -> Conv(7, ngf) -> Norm -> ReLU
                 -> 2x [Conv(3, stride=2) -> Norm -> ReLU]
        Transform: 9x ResnetBlock
        Decoder: 2x [ConvTranspose(3, stride=2) -> Norm -> ReLU]
                 -> ReflectionPad(3) -> Conv(7, out_ch) -> Tanh
    """

    NUM_DOWNSAMPLES: int = 2
    NUM_RESNET_BLOCKS: int = 9

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the ResNet generator from model configuration.

        Args:
            config: Model configuration dataclass with architecture hyperparameters.
        """
        super().__init__()

        input_channels: int = config.input_channels
        output_channels: int = config.output_channels
        ngf: int = config.ngf
        norm_layer = get_norm_layer(config.norm_type)

        # --- Encoder: initial convolution ---
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, ngf, kernel_size=7, padding=0, bias=True),
            norm_layer(ngf),
            nn.ReLU(inplace=True),
        ]

        # --- Encoder: downsampling ---
        for i in range(self.NUM_DOWNSAMPLES):
            mult = 2 ** i
            in_ch = ngf * mult
            out_ch = ngf * mult * 2
            model += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=True),
                norm_layer(out_ch),
                nn.ReLU(inplace=True),
            ]

        # --- Transformer: residual blocks ---
        mult = 2 ** self.NUM_DOWNSAMPLES
        for _ in range(self.NUM_RESNET_BLOCKS):
            model += [ResnetBlock(ngf * mult, norm_layer)]

        # --- Decoder: upsampling ---
        for i in range(self.NUM_DOWNSAMPLES):
            mult = 2 ** (self.NUM_DOWNSAMPLES - i)
            in_ch = ngf * mult
            out_ch = ngf * mult // 2
            model += [
                nn.ConvTranspose2d(
                    in_ch, out_ch,
                    kernel_size=3, stride=2, padding=1, output_padding=1, bias=True,
                ),
                norm_layer(out_ch),
                nn.ReLU(inplace=True),
            ]

        # --- Decoder: output convolution ---
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Translate input image through the generator.

        Args:
            x: Input tensor of shape (B, C_in, 256, 256).

        Returns:
            Translated image tensor of shape (B, C_out, 256, 256) in [-1, 1].
        """
        return self.model(x)
