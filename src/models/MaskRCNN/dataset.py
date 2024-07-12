import os

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class SpineDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = sorted(
            [
                f
                for f in os.listdir(os.path.join(root, "input_images"))
                if os.path.isfile(os.path.join(root, "input_images", f))
            ]
        )
        self.masks = sorted(
            [
                f
                for f in os.listdir(os.path.join(root, "spine_images"))
                if os.path.isfile(os.path.join(root, "spine_images", f))
            ]
        )

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "input_images", self.imgs[idx])
        mask_path = os.path.join(self.root, "spine_images", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        img = np.array(img)
        mask = np.array(mask)

        if self.transforms:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        obj_ids = np.unique(mask)[1:]  # Remove background ID

        boxes, masks = [], []
        for obj_id in obj_ids:
            mask_obj = mask == obj_id
            pos = np.where(mask_obj)
            if pos[0].size > 0 and pos[1].size > 0:  # Ensure mask is not empty
                xmin, xmax = np.min(pos[1]), np.max(pos[1])
                ymin, ymax = np.min(pos[0]), np.max(pos[0])
                if xmax > xmin and ymax > ymin:  # Ensure bounding box is valid
                    boxes.append([xmin, ymin, xmax, ymax])
                    masks.append(mask_obj)

        if len(boxes) == 0:
            # Handling case where no objects are found
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones(
                (len(boxes),), dtype=torch.int64
            )  # Assuming all objects are of class '1'
            masks = np.array(masks, dtype=np.uint8)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
        }

        return img, target

    def __len__(self):
        return len(self.imgs)
