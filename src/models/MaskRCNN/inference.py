import os
import argparse
import torch
from tqdm import tqdm
import numpy as np
from torchvision.transforms import functional as F
from PIL import Image
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2
from model import get_model_instance_segmentation
import cv2


def load_model(model_path, num_classes, device):
    """
    Load and initialize the instance segmentation model.

    Parameters:
    - model_path (str): Path to the trained model.
    - num_classes (int): Number of classes for the model.
    - device (torch.device): Device to load the model on (CPU or GPU).

    Returns:
    - model (torch.nn.Module): The loaded and initialized model.
    """
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, device):
    """
    Preprocess the image for model inference.

    Parameters:
    - image_path (str): Path to the input image.
    - device (torch.device): Device to move the tensor to (CPU or GPU).

    Returns:
    - img_tensor (torch.Tensor): Preprocessed image tensor.
    """
    # Define the transformation pipeline for inference
    transform = Compose([
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # Load and preprocess the input image using Albumentations
    img = Image.open(image_path).convert("RGB")
    img = np.array(img)
    transformed = transform(image=img)
    img_tensor = transformed["image"].unsqueeze(0).to(device)
    return img_tensor


def run_inference(model, device, image_path, threshold=0.5):
    """
    Run inference on the given image using the model.

    Parameters:
    - model (torch.nn.Module): The model used for inference.
    - device (torch.device): Device to run the inference on (CPU or GPU).
    - image_path (str): Path to the input image.
    - threshold (float): Threshold for filtering predictions.

    Returns:
    - pred_boxes (np.ndarray): Predicted bounding boxes.
    - pred_masks (np.ndarray): Predicted masks.
    - pred_scores (np.ndarray): Predicted scores.
    """
    img_tensor = preprocess_image(image_path, device)

    # Perform inference
    with torch.no_grad():
        prediction = model(img_tensor)

    # Process the predictions
    pred_boxes = prediction[0]["boxes"].cpu().numpy()
    pred_scores = prediction[0]["scores"].cpu().numpy() # Model's confidence for each prediction
    pred_masks = prediction[0]["masks"].cpu().numpy()

    # Filter predictions by threshold
    pred_boxes = pred_boxes[pred_scores >= threshold]
    pred_masks = pred_masks[pred_scores >= threshold]
    pred_scores = pred_scores[pred_scores >= threshold]

    return pred_boxes, pred_masks, pred_scores


def visualize_results(img, boxes, masks, scores, threshold=0.5):
    """
    Visualize the results by drawing bounding boxes and applying masks.

    Parameters:
    - img (PIL.Image): The original image.
    - boxes (np.ndarray): Bounding boxes to draw.
    - masks (np.ndarray): Masks to apply.
    - scores (np.ndarray): Confidence scores.
    - threshold (float): Threshold for displaying the results.

    Returns:
    - result_img (PIL.Image): Image with visualized results.
    """
    img_np = np.array(img)

    for i, box in enumerate(boxes):
        if scores[i] >= threshold:
            xmin, ymin, xmax, ymax = box.astype(int)
            # Draw the bounding box
            cv2.rectangle(img_np, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            # Apply the mask on i-th prediction and 0-th channel (single-channel masks) 
            mask = masks[i, 0] > 0
            img_np[mask] = [255, 0, 0]  # Red color for mask

    return Image.fromarray(img_np)


def main():
    """
    Main function to run inference on the dataset and save the results.
    """
    # Define command line arguments
    parser = argparse.ArgumentParser(description="Inference for Spine Segmentation")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model"
    )
    parser.add_argument(
        "--Validation_Folder",
        type=str,
        required=True,
        help="Path to the validation folder",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output images and metrics",
    )
    args = parser.parse_args()

    # Set device and number of classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2  # Background and spine

    # Load the model
    model = load_model(args.model_path, num_classes, device)

    # Initialize lists for storing results
    predictions_list = []

    # Prepare input and output folders
    input_folder = os.path.join(args.Validation_Folder, "input_images")
    output_folder = args.output_path
    os.makedirs(output_folder, exist_ok=True)

    # Sort the files based on their numerical suffixes
    input_images = sorted([f for f in os.listdir(input_folder) if f.endswith(".png")])

    # Add tqdm to display the progress bar
    for input_image_name in tqdm(input_images, desc="Processing images"):
        image_path = os.path.join(input_folder, input_image_name)
        img = Image.open(image_path).convert("RGB")

        # Run inference
        boxes, masks, scores = run_inference(
            model, device, image_path, threshold=0.5
        )
        # Visualize and save results
        predicted_mask = visualize_results(img, boxes, masks, scores, threshold=0.5)
        predictions_list.append(predicted_mask)

        # Save prediction mask
        predicted_mask.save(os.path.join(output_folder, f"pred_{input_image_name}"))


if __name__ == "__main__":
    main()

