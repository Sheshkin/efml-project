import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from PIL import Image


class SegmentationTransform:
    def __init__(self, image_size=256, augment=False):
        self.image_size = image_size
        self.augment = augment

    def __call__(self, image, mask):
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        if self.augment:
            if np.random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.random() > 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        img_t = transforms.ToTensor()(image)
        img_t = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])(img_t)

        mask_arr = np.array(mask, dtype=np.int64)
        binary_mask = (mask_arr == 1).astype(np.float32)
        mask_t = torch.from_numpy(binary_mask).unsqueeze(0)

        return img_t, mask_t


class PetSegmentationDataset(Dataset):
    def __init__(self, root, split="trainval", image_size=256, augment=False, download=True):
        self.transform = SegmentationTransform(image_size=image_size, augment=augment)
        self.base = OxfordIIITPet(
            root=root,
            split=split,
            target_types="segmentation",
            download=download,
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, mask = self.base[idx]
        return self.transform(image, mask)


def build_dataloaders(data_root, image_size=256, batch_size=8, val_fraction=0.15, num_workers=0, seed=42):
    full_train = PetSegmentationDataset(root=data_root, split="trainval",
                                        image_size=image_size, augment=True)
    n_val = int(len(full_train) * val_fraction)
    n_train = len(full_train) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=generator)

    val_ds.dataset = PetSegmentationDataset(root=data_root, split="trainval",
                                             image_size=image_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False)
    return train_loader, val_loader


def build_test_loader(data_root, image_size=256, batch_size=8, num_workers=0):
    test_ds = PetSegmentationDataset(root=data_root, split="test",
                                     image_size=image_size, augment=False)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=False)
