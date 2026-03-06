import matplotlib.pyplot as plt
from dataloader.dataloader import load_shipsnet

# This file is just to check that the labels are correct and that the data looks right.
def main():
    train_loader, _, _ = load_shipsnet()
    x, y = next(iter(train_loader))

    # Show 12 images with labels so you can see which is ship
    fig, axes = plt.subplots(3, 4, figsize=(10, 7))
    axes = axes.flatten()

    for i in range(12):
        img = x[i].clamp(0, 1).permute(1, 2, 0)
        axes[i].imshow(img)
        axes[i].set_title(f"label={y[i].item()}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()