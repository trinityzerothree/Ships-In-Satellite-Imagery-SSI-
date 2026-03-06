import matplotlib.pyplot as plt
import torch
from dataloader import load_shipsnet, rotate_4_directions


def display_samples():
    train_loader, val_loader, test_loader = load_shipsnet(
        use_rotation_augmentation=True, num_workers=0
    )

    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")

    num_samples = 4
    fig, axes = plt.subplots(num_samples, 5, figsize=(12, 10))
    fig.suptitle("Sample Images with 4-Direction Rotation Augmentation", fontsize=14)

    for i in range(num_samples):
        original = images[i]
        
        axes[i, 0].imshow(original.permute(1, 2, 0))
        axes[i, 0].set_title("Original" if i == 0 else "")
        axes[i, 0].axis("off")

        rotated = rotate_4_directions(original)
        for j in range(4):
            axes[i, j + 1].imshow(rotated[j].permute(1, 2, 0))
            if i == 0:
                axes[i, j + 1].set_title(f"Rotated {j*90}°")
            axes[i, j + 1].axis("off")

    plt.tight_layout()
    plt.savefig("sample_augmented_images.png", dpi=150)
    print("Saved sample_augmented_images.png")


if __name__ == "__main__":
    display_samples()
