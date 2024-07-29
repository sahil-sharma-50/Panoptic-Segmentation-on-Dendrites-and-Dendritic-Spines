import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50


def get_model():
    """
    Initialize the FCN-ResNet50 model and modify the classifier to output a single channel for binary segmentation.
    """
    model = fcn_resnet50(weights=None, progress=True)
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)
    return model


def load_image(image_path, transform=None):
    """
    Load and transform an image.

    Parameters:
    - image_path (str): Path to the input image.
    - transform (callable, optional): Transformation to apply to the image.

    Returns:
    - image (Tensor): Transformed image tensor.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    if transform:
        image = transform(image)
    return image


def load_mask(mask_path, transform=None):
    """
    Load and process a mask.

    Parameters:
    - mask_path (str): Path to the mask image.
    - transform (callable, optional): Transformation to apply to the mask.

    Returns:
    - mask (np.ndarray): Binary mask array.
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    mask = Image.open(mask_path).convert("L")
    if transform:
        mask = transform(mask)
    mask = np.array(mask)
    mask = (mask > 0).astype(np.uint8)  # Ensure binary mask
    return mask


def calculate_metrics(pred_mask, true_mask):
    """
    Calculate IoU, Precision, Recall, and F1 Score.

    Parameters:
    - pred_mask (np.ndarray): Predicted mask.
    - true_mask (np.ndarray): Ground truth mask.

    Returns:
    - (tuple): IoU, Precision, Recall, F1 Score.
    """
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    # Handle divisible by zero, if union is zero make iou zero
    iou = intersection / union if union != 0 else 0
    precision = intersection / pred_mask.sum() if pred_mask.sum() != 0 else 0
    recall = intersection / true_mask.sum() if true_mask.sum() != 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) != 0
        else 0
    )
    return iou, precision, recall, f1


def dice_loss(pred_mask, true_mask, smooth=1):
    """
    Calculate the Dice Loss.

    Parameters:
    - pred_mask (np.ndarray): Predicted mask.
    - true_mask (np.ndarray): Ground truth mask.
    - smooth (float): Smoothing constant to avoid division by zero.

    Returns:
    - (float): Dice loss.
    """
    pred_flat = pred_mask.flatten()
    true_flat = true_mask.flatten()
    intersection = (pred_flat * true_flat).sum()
    return 1 - (
        (2.0 * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)
    )


def main():
    # Define paths
    input_folder = "Dataset/DeepD3_Validation/input_images"
    dendrite_folder = "Dataset/DeepD3_Validation/dendrite_images"

    # Define transformations
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)

    # Load the trained model weights
    model.load_state_dict(
        torch.load("dendrite_model.pt", map_location=device), strict=False
    )
    model.eval()  # Set the model to evaluation mode

    # Load image and mask paths
    input_images = sorted([f for f in os.listdir(input_folder) if f.endswith(".png")])
    dendrite_images = sorted(
        [f for f in os.listdir(dendrite_folder) if f.endswith(".png")]
    )

    # Initialize lists to store evaluation metrics
    ious, precisions, recalls, f1s, dice_losses = [], [], [], [], []

    # Perform inference and calculate metrics
    with torch.no_grad():
        for input_image, dendrite_image in zip(input_images, dendrite_images):
            # Load and preprocess images and masks
            input_image_path = os.path.join(input_folder, input_image)
            dendrite_image_path = os.path.join(dendrite_folder, dendrite_image)

            image = load_image(input_image_path, transform).unsqueeze(0).to(device)
            true_mask = load_mask(dendrite_image_path, transform)

            # Perform inference
            output = model(image)["out"]
            pred_mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

            # Calculate metrics
            iou, precision, recall, f1 = calculate_metrics(pred_mask, true_mask)
            dice = dice_loss(pred_mask, true_mask)

            # Append metrics to respective lists
            ious.append(iou)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            dice_losses.append(dice)

    # Compute and print average metrics
    mean_iou = np.mean(ious)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1s)
    mean_dice_loss = np.mean(dice_losses)

    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}")
    print(f"Mean F1 Score: {mean_f1:.4f}")
    print(f"Mean Dice Loss: {mean_dice_loss:.4f}")


if __name__ == "__main__":
    main()
