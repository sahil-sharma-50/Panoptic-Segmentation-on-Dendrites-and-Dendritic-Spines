import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights


def get_model_instance_segmentation(num_classes):
    """
    Initialize and modify the Mask R-CNN model for instance segmentation.

    Parameters:
    - num_classes (int): Number of output classes including the background.

    Returns:
    - model (nn.Module): Modified Mask R-CNN model.
    """
    # Load pre-trained weights
    weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
    model = maskrcnn_resnet50_fpn(weights=weights)

    # Modify the box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Modify the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 512
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

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


def load_mask(mask_path):
    """
    Load and process a mask.

    Parameters:
    - mask_path (str): Path to the mask image.

    Returns:
    - mask (np.ndarray): Binary mask array.
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask)
    mask = (mask > 0).astype(np.uint8)  # Ensure binary mask
    return mask


def calculate_metrics(pred_mask, true_mask):
    """
    Calculate IoU, Precision, and Recall.

    Parameters:
    - pred_mask (np.ndarray): Predicted mask.
    - true_mask (np.ndarray): Ground truth mask.

    Returns:
    - (tuple): IoU, Precision, Recall.
    """
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    # Handle divisible by zero, if union is zero make iou zero
    iou = intersection / union if union != 0 else 0
    precision = intersection / pred_mask.sum() if pred_mask.sum() != 0 else 0
    recall = intersection / true_mask.sum() if true_mask.sum() != 0 else 0
    return iou, precision, recall


def main():
    # Define paths
    input_folder = "Dataset/DeepD3_Validation/input_images"
    spine_folder = "Dataset/DeepD3_Validation/spine_images"

    # Define transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model_instance_segmentation(num_classes=2).to(device)

    # Load the trained model weights
    model.load_state_dict(
        torch.load("spines_model.pt", map_location=device), strict=False
    )
    model.eval()  # Set the model to evaluation mode

    # Load image and mask paths
    input_images = sorted([f for f in os.listdir(input_folder) if f.endswith(".png")])
    spine_images = sorted([f for f in os.listdir(spine_folder) if f.endswith(".png")])

    ious, precisions, recalls = [], [], []

    # Perform inference and calculate metrics
    with torch.no_grad():
        for input_image, spine_image in zip(input_images, spine_images):
            # Load and preprocess images and masks
            input_image_path = os.path.join(input_folder, input_image)
            spine_image_path = os.path.join(spine_folder, spine_image)

            image = load_image(input_image_path, transform).unsqueeze(0).to(device)
            true_mask = load_mask(spine_image_path)

            # Perform inference
            output = model(image)
            pred_masks = output[0]["masks"] > 0.5

            # Check if number of predicted masks is zero
            if pred_masks.shape[0] == 0:
                continue

            pred_mask = pred_masks.squeeze().cpu().numpy().astype(np.uint8)

            # combining multiple predicted masks into a single mask.
            if pred_mask.ndim > 2:
                pred_mask = np.max(pred_mask, axis=0)

            # Ensure the predicted mask is the same size as the true mask
            pred_mask_resized = cv2.resize(
                pred_mask,
                (true_mask.shape[1], true_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

            # Calculate metrics
            iou, precision, recall = calculate_metrics(pred_mask_resized, true_mask)
            ious.append(iou)
            precisions.append(precision)
            recalls.append(recall)

    # Compute and print average metrics
    mean_iou = np.mean(ious)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)

    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}")


if __name__ == "__main__":
    main()
