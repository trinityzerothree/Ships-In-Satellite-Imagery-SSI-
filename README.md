# Ships in Satellite Imagery

Project for binary classification of ships in 80x80 RGB satellite images using PyTorch.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The dataset auto-downloads from Kaggle on first run. You'll be prompted for your Kaggle username and API key.

## Training

```bash
python -m src.train
```

This trains a CNN for 10 epochs, evaluates on a held-out test set, and saves the best model to `best_cnn.pt`.

## Data Augmentation

Enable rotation augmentation for training:

```python
train_loader, val_loader, test_loader = load_shipsnet(use_rotation_augmentation=True)
```

The `rotate_4_directions()` function generates all 4 rotations (0°, 90°, 180°, 270°). Run `python display_samples.py` to visualize augmented examples.
