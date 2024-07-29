import os
from PIL import Image
from torch.utils.data import Dataset


class DendritesDataset(Dataset):
    """
    Custom dataset for loading dendrite images and their corresponding masks.

    Attributes:
    - root (str): Root directory containing 'input_images' and 'dendrite_images' subdirectories.
    - transforms (callable, optional): Optional transforms to be applied on a sample.
    - images (list): List of image filenames.
    - masks (list): List of mask filenames.
    """

    def __init__(self, root, transforms=None):
        """
        Initialize the dataset with the directory path and optional transforms.

        Parameters:
        - root (str): Root directory containing the dataset.
        - transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.transforms = transforms

        # Get sorted list of image and mask filenames
        self.images = sorted(
            [
                f
                for f in os.listdir(os.path.join(root, "input_images"))
                if os.path.isfile(os.path.join(root, "input_images", f))
            ]
        )
        self.masks = sorted(
            [
                f
                for f in os.listdir(os.path.join(root, "dendrite_images"))
                if os.path.isfile(os.path.join(root, "dendrite_images", f))
            ]
        )

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
        - (int): Number of samples.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset at the specified index.

        Parameters:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - image (PIL Image or transformed tensor): The input image.
        - mask (PIL Image or transformed tensor): The corresponding mask image.
        """
        img_path = os.path.join(self.root, "input_images", self.images[idx])
        mask_path = os.path.join(self.root, "dendrite_images", self.masks[idx])

        image = Image.open(img_path).convert("RGB")  # Load and convert image to RGB
        mask = Image.open(mask_path).convert("L")  # Load and convert mask to grayscale

        if self.transforms:
            # Apply transformations
            image = self.transforms(image)
            mask = self.transforms(mask)
            mask = (mask > 0).float()  # Convert mask to binary (0 or 1)

        return image, mask
