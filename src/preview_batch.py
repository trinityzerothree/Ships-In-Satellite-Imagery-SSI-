import matplotlib.pyplot as plt
import torch

from dataloader.dataloader import load_shipsnet
from src.augmentations import train_transform

# This file is just to check that the augmentations are working and to see some examples of augmented images.
def main():
    train_loader, val_loader, test_loader = load_shipsnet(transform=train_transform())

    images, labels = next(iter(train_loader))
    print("images:", images.shape, "labels:", labels.shape)

    # show 8 augmented images
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    axes = axes.flatten()

    for i in range(8):
        img = images[i].clamp(0, 1)
        axes[i].imshow(img.permute(1, 2, 0))
        axes[i].set_title(f"label={labels[i].item()}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()