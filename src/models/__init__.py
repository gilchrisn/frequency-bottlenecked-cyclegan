"""Model factory for CycleGAN generators and discriminators.

Provides registry-based creation with automatic weight initialization.
Extend by adding new entries to the GENERATORS or DISCRIMINATORS dicts.
"""

from typing import Dict, Type

import torch.nn as nn

from src.config import ModelConfig
from src.models.discriminator import PatchGANDiscriminator
from src.models.generator import ResnetGenerator, init_weights

# ---------------------------------------------------------------------------
# Model registries (Open-Closed Principle: add new class, never modify existing)
# ---------------------------------------------------------------------------

GENERATORS: Dict[str, Type[nn.Module]] = {
    "resnet_9blocks": ResnetGenerator,
}

DISCRIMINATORS: Dict[str, Type[nn.Module]] = {
    "patchgan_70": PatchGANDiscriminator,
}


def create_generator(config: ModelConfig) -> nn.Module:
    """Create and initialize a generator from configuration.

    Args:
        config: Model configuration specifying generator and init params.

    Returns:
        Initialized generator network.

    Raises:
        KeyError: If config.generator is not in the GENERATORS registry.
    """
    if config.generator not in GENERATORS:
        raise KeyError(
            f"Unknown generator type '{config.generator}'. "
            f"Available: {list(GENERATORS.keys())}"
        )
    net = GENERATORS[config.generator](config)
    init_weights(net, init_type=config.init_type, init_gain=config.init_gain)
    return net


def create_discriminator(config: ModelConfig) -> nn.Module:
    """Create and initialize a discriminator from configuration.

    Args:
        config: Model configuration specifying discriminator and init params.

    Returns:
        Initialized discriminator network.

    Raises:
        KeyError: If config.discriminator is not in the DISCRIMINATORS registry.
    """
    if config.discriminator not in DISCRIMINATORS:
        raise KeyError(
            f"Unknown discriminator type '{config.discriminator}'. "
            f"Available: {list(DISCRIMINATORS.keys())}"
        )
    net = DISCRIMINATORS[config.discriminator](config)
    init_weights(net, init_type=config.init_type, init_gain=config.init_gain)
    return net
