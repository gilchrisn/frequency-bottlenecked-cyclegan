"""Data module: dataset creation and dataloader factories."""

from torch.utils.data import DataLoader

from src.config import DataConfig, TrainConfig
from src.data.brats_dataset import BraTSDataset
from src.data.transforms import get_train_transform, get_val_transform


def create_dataset(config: DataConfig, split: str = "train") -> BraTSDataset:
    """Factory function to create a BraTS dataset with appropriate transforms.

    Args:
        config: Data configuration object.
        split: One of "train", "val", or "test".

    Returns:
        BraTSDataset instance with the correct transform for the split.
    """
    if split == "train":
        transform = get_train_transform(config)
    else:
        transform = get_val_transform(config)

    return BraTSDataset(config=config, split=split, transform=transform)


def create_dataloader(
    dataset: BraTSDataset,
    data_config: DataConfig,
    train_config: TrainConfig,
    split: str = "train",
) -> DataLoader:
    """Factory function to wrap a dataset in a DataLoader.

    Args:
        dataset: The BraTS dataset to wrap.
        data_config: Data configuration (for num_workers).
        train_config: Training configuration (for batch_size).
        split: Split name; determines shuffle behavior.

    Returns:
        DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=(split == "train"),
        num_workers=data_config.num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
