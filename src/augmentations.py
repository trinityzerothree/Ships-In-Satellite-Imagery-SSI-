import torchvision.transforms.v2 as T

#basic augmentations for training
def train_transform():
    return T.Compose([
        T.RandomRotation(15),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandomGrayscale(p=0.2),
        T.RandomResizedCrop(size=(80, 80), scale=(0.7, 1.0)),
    ])