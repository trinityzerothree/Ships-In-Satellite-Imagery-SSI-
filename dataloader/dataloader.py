import json
import os
import numpy as np
import opendatasets as od
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import random

KAGGLE_URL = "https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
JSON_FILENAME = "shipsnet.json"
BATCH_SIZE = 32


def rotate_4_directions(image: torch.Tensor) -> torch.Tensor:
    """Rotate image in all 4 directions (0°, 90°, 180°, 270°) and stack them.
    
    Args:
        image: Input image tensor of shape (C, H, W).
        
    Returns:
        Stacked tensor of shape (4, C, H, W) containing all 4 rotations.
    """
    rotations = []
    for k in range(4):
        rotated = torch.rot90(image, k, dims=[1, 2])
        rotations.append(rotated)
    return torch.stack(rotations)

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
        image = np.array(self.data[idx], dtype=np.float32).reshape(3, 80, 80) / 255.0
        label = self.labels[idx]

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
    use_rotation_augmentation: bool = False,
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
        transform: Optional transform applied to each image tensor.
        use_rotation_augmentation: If True, apply random 90° rotation to training images.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    json_path = _download_dataset(data_dir)

    with open(json_path) as f:
        raw = json.load(f)

    data = raw["data"]
    labels = raw["labels"]

    if use_rotation_augmentation:
        base_transform = transform
        def augment_transform(image):
            k = random.randint(0, 3)
            image = torch.rot90(image, k, dims=[1, 2])
            if base_transform:
                image = base_transform(image)
            return image
        transform = augment_transform

    dataset = ShipsDataset(data, labels, transform=transform)
    indices = list(range(len(dataset)))

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

    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_shipsnet()

    images, labels = next(iter(train_loader))
    print("Info on the first batch:")
    print(f"Shape: {images.shape}, Labels: {labels}")
