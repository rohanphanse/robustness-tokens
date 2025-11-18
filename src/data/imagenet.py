import os

import dotenv
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from data.transforms import default_transform

def get_loaders(
    batch_size,
    num_workers=4,
    train_transform=default_transform,
    val_transform=default_transform,
):
    dotenv.load_dotenv()
    IMAGENET_DIR = os.getenv("IMAGENET_DIR")

    assert IMAGENET_DIR is not None, "IMAGENET_DIR is not set in .env"

    train_dir = os.path.join(IMAGENET_DIR, "train")
    val_dir = os.path.join(IMAGENET_DIR, "val")

    train_set = ImageFolder(root=train_dir, transform=train_transform)
    val_set = ImageFolder(root=val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader
