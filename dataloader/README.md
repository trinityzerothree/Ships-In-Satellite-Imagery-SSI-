# Ships in Satellite Imagery Dataloader

Dataloader for binary classification of ships in 80x80 RGB satellite images using PyTorch.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Kaggle credentials

The dataset auto-downloads from Kaggle on first run. You'll be prompted for your Kaggle username and API key.

## Usage

```python
from dataloader import load_shipsnet

train_loader, val_loader, test_loader = load_shipsnet()

for images, labels in train_loader:
    # images: (B, 3, 80, 80), labels: (B,)
    ...
```

Or run directly:

```bash
python dataloader.py
```

## Data Augmentation

Enable rotation augmentation for training:

```python
train_loader, val_loader, test_loader = load_shipsnet(use_rotation_augmentation=True)
```

The `rotate_4_directions()` function generates all 4 rotations (0°, 90°, 180°, 270°). Run `python display_samples.py` to visualize augmented examples.
