import json
import os
import numpy as np
import opendatasets as od
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

KAGGLE_URL = "https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
JSON_FILENAME = "shipsnet.json"
BATCH_SIZE = 32


def _download_dataset(data_dir: str) -> str:
    """Download the dataset from Kaggle if not already present. Returns path to shipsnet.json."""
    json_path = os.path.join(data_dir, JSON_FILENAME)
    if os.path.exists(json_path):
        return json_path

    od.download(KAGGLE_URL, data_dir=data_dir)

    # opendatasets extracts into a subdirectory named after the dataset
    extracted = os.path.join(data_dir, "ships-in-satellite-imagery", JSON_FILENAME)
    if os.path.exists(extracted) and not os.path.exists(json_path):
        os.rename(extracted, json_path)

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Downloaded dataset but could not find {JSON_FILENAME} in {data_dir}"
        )
    
    return json_path

class ShipsDataset(Dataset):
    """Dataset for Ships in Satellite Imagery (shipsnet.json)."""
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Reshape flat pixel array to (3, 80, 80) and normalize to [0, 1]
        image = self.data[idx].reshape(3, 80, 80) / 255.0
        label = int(self.labels[idx])

        image = torch.from_numpy(image)
        if self.transform:
            image = self.transform(image)

        return image, label

def load_shipsnet(
    data_dir: str = DATA_DIR,
    batch_size: int = BATCH_SIZE,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
    transform=None,
    eval_transform=None,
    offline_augmentation: bool = False,
    offline_rotations: bool = True,
    offline_flips: bool = True,
):
    """Download (if needed) and load shipsnet, returning train/val/test DataLoaders.

    Args:
        data_dir: Directory to store/find the dataset.
        batch_size: Batch size for all loaders.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        seed: Random seed for reproducible splits.
        num_workers: Workers for DataLoader.
        transform: Optional transform applied to training images.
        eval_transform: Optional transform applied to val/test images. If None, no transform is applied.
        offline_augmentation: If True, expand training data with deterministic augmentations.
        offline_rotations: Include 90°/180°/270° rotations in offline augmentation.
        offline_flips: Include horizontal/vertical flips in offline augmentation.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    json_path = _download_dataset(data_dir)

    with open(json_path) as f:
        raw = json.load(f)

    data = np.array(raw["data"], dtype=np.float32)
    labels = np.array(raw["labels"], dtype=np.int64)
    del raw  # free the parsed JSON

    indices = list(range(len(data)))

    # First split: train vs (val + test), stratified by label
    train_idx, valtest_idx = train_test_split(
        indices, test_size=val_ratio + test_ratio, random_state=seed, stratify=labels
    )

    # Second split: val vs test
    valtest_labels = [labels[i] for i in valtest_idx]
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        valtest_idx, test_size=relative_test, random_state=seed, stratify=valtest_labels
    )

    # Build training data (with optional offline augmentation)
    train_data = data[train_idx]
    train_labels = labels[train_idx]
    if offline_augmentation:
        from src.augmentations import expand_with_augmentations
        train_data, train_labels = expand_with_augmentations(
            train_data, train_labels, rotations=offline_rotations, flips=offline_flips
        )
    train_dataset = ShipsDataset(train_data, train_labels, transform=transform)

    # Build eval datasets from original data only
    val_data = data[val_idx]
    val_labels = labels[val_idx]
    test_data = data[test_idx]
    test_labels = labels[test_idx]
    val_dataset = ShipsDataset(val_data, val_labels, transform=eval_transform)
    test_dataset = ShipsDataset(test_data, test_labels, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_shipsnet()

    images, labels = next(iter(train_loader))
    print("Info on the first batch:")
    print(f"Shape: {images.shape}, Labels: {labels}")
