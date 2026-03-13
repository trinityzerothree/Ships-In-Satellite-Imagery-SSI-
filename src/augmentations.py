import numpy as np
import torch
import torchvision.transforms.v2 as T
from tqdm import tqdm

# On-the-fly augmentations (applied randomly each time an image is loaded)
def train_transform(
    rotation=True,
    horizontal_flip=True,
    vertical_flip=True,
    color_jitter=True,
    grayscale=True,
    random_crop=True,
):
    transforms = []
    if rotation:
        transforms.append(T.RandomRotation(15))
    if horizontal_flip:
        transforms.append(T.RandomHorizontalFlip(0.5))
    if vertical_flip:
        transforms.append(T.RandomVerticalFlip(0.5))
    if color_jitter:
        transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
    if grayscale:
        transforms.append(T.RandomGrayscale(p=0.2))
    if random_crop:
        transforms.append(T.RandomResizedCrop(size=(80, 80), scale=(0.7, 1.0)))
    return T.Compose(transforms)

# Offline augmentations (generate new samples added to the dataset)
def rotate_4_directions(image: torch.Tensor) -> torch.Tensor:
    """Rotate image in all 4 directions (0, 90, 180, 270) and stack them.

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

def expand_with_augmentations(data, labels, rotations=True, flips=True):
    """Generate new samples using deterministic augmentations (rotations and flips).

    Args:
        data: List of flat pixel arrays.
        labels: List of labels.
        rotations: If True, add 90, 180, 270 rotated copies.
        flips: If True, add horizontally and vertically flipped copies.

    Returns:
        Tuple of (expanded_data, expanded_labels) as numpy arrays.
    """
    images = np.asarray(data, dtype=np.float32).reshape(-1, 3, 80, 80)
    expanded_images = [images]
    expanded_labels = [np.array(labels)]

    steps = []
    if rotations:
        steps += [(k, f"Rotating {k*90}°") for k in [1, 2, 3]]
    if flips:
        steps += [("hflip", "Horizontal flip"), ("vflip", "Vertical flip")]

    for step, desc in tqdm(steps, desc="Offline augmentation"):
        if isinstance(step, int):
            expanded_images.append(np.rot90(images, step, axes=(2, 3)))
        elif step == "hflip":
            expanded_images.append(images[:, :, :, ::-1])
        elif step == "vflip":
            expanded_images.append(images[:, :, ::-1, :])
        expanded_labels.append(np.array(labels))

    all_images = np.concatenate(expanded_images, axis=0)
    all_labels = np.concatenate(expanded_labels, axis=0)
    # Flatten back to match ShipsDataset expected format
    all_data = np.ascontiguousarray(all_images.reshape(len(all_labels), -1))
    return all_data, all_labels
