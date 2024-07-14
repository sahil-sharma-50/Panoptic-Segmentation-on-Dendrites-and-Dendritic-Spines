import os
from PIL import Image
from torch.utils.data import Dataset

class DendritesDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        
        self.images = sorted([f for f in os.listdir(os.path.join(root, "input_images")) if os.path.isfile(os.path.join(root, "input_images", f))])
        self.masks = sorted([f for f in os.listdir(os.path.join(root, "dendrite_images")) if os.path.isfile(os.path.join(root, "dendrite_images", f))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "input_images", self.images[idx])
        mask_path = os.path.join(self.root, "dendrite_images", self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transforms:
            image = self.transforms(image)
            mask = self.transforms(mask)
            mask = (mask > 0).float()
        return image, mask
