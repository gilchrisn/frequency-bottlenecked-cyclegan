"""Configuration system for FB-CycleGAN experiments.

All hyperparameters are centralized here. No magic numbers in other files.
Presets define reproducible experiment configurations.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ModelConfig:
    """Generator and discriminator architecture settings."""

    generator: str = "resnet_9blocks"
    discriminator: str = "patchgan_70"
    input_channels: int = 1
    output_channels: int = 1
    ngf: int = 64
    ndf: int = 64
    norm_type: str = "instance"
    no_dropout: bool = True
    init_type: str = "normal"
    init_gain: float = 0.02


@dataclass
class LossConfig:
    """Loss function weights and frequency-bottleneck settings."""

    gan_mode: str = "lsgan"
    lambda_cycle: float = 10.0
    lambda_identity: float = 0.5
    use_frequency_bottleneck: bool = False
    blur_kernel_size: int = 5
    blur_sigma: float = 1.0


@dataclass
class TrainConfig:
    """Training hyperparameters and schedule."""

    epochs: int = 200
    batch_size: int = 8
    compile_models: bool = True
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    lr_policy: str = "linear"
    lr_decay_start: int = 100
    pool_size: int = 50
    save_freq: int = 5
    log_freq: int = 100
    val_freq: int = 1


@dataclass
class DataConfig:
    """Dataset loading and augmentation settings."""

    image_size: int = 256
    crop_size: int = 256
    load_size: int = 286
    flip: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    tumor_area_threshold: float = 0.05
    min_brain_area: int = 1000
    mri_sequence: str = "flair"


@dataclass
class EvalConfig:
    """Evaluation metric settings."""

    fid_num_samples: int = 500
    perturbation_sigma: float = 0.01


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration aggregating all sub-configs."""

    name: str = "experiment"
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 42
    device: str = "cuda"
    use_wandb: bool = True
    output_dir: str = "outputs"


def _make_fb_preset(sigma: float) -> ExperimentConfig:
    """Create a frequency-bottleneck preset with the given sigma.

    Args:
        sigma: Standard deviation of the Gaussian blur kernel.

    Returns:
        ExperimentConfig with frequency bottleneck enabled.
    """
    kernel_size = max(3, int(sigma * 4) | 1)
    return ExperimentConfig(
        name=f"fb_cyclegan_sigma{sigma}",
        loss=LossConfig(
            use_frequency_bottleneck=True,
            blur_kernel_size=kernel_size,
            blur_sigma=sigma,
        ),
    )


PRESETS: Dict[str, ExperimentConfig] = {
    "baseline_cyclegan": ExperimentConfig(
        name="baseline_cyclegan",
        loss=LossConfig(use_frequency_bottleneck=False),
    ),
    "fb_cyclegan_sigma0.5": _make_fb_preset(0.5),
    "fb_cyclegan_sigma1": _make_fb_preset(1.0),
    "fb_cyclegan_sigma1.5": _make_fb_preset(1.5),
    "fb_cyclegan_sigma2": _make_fb_preset(2.0),
    "fb_cyclegan_sigma3": _make_fb_preset(3.0),
}


def get_config(preset: str) -> ExperimentConfig:
    """Retrieve an experiment configuration by preset name.

    Args:
        preset: Key into the PRESETS dictionary.

    Returns:
        A copy of the matching ExperimentConfig.

    Raises:
        KeyError: If preset is not found in PRESETS.
    """
    if preset not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise KeyError(f"Unknown preset '{preset}'. Available: {available}")

    import copy
    return copy.deepcopy(PRESETS[preset])
