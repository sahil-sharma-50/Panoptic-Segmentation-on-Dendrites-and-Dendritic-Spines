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
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, device):
    # Define the transformation pipeline for inference
    transform = Compose(
        [Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
    )

    # Load and preprocess the input image using Albumentations
    img = Image.open(image_path).convert("RGB")
    img = np.array(img)
    transformed = transform(image=img)
    img_tensor = transformed["image"].unsqueeze(0).to(device)
    return img_tensor


def run_inference(model, device, image_path, threshold=0.5):
    img_tensor = preprocess_image(image_path, device)

    # Perform inference
    with torch.no_grad():
        prediction = model(img_tensor)

    # Process the predictions
    pred_boxes = prediction[0]["boxes"].cpu().numpy()
    pred_scores = prediction[0]["scores"].cpu().numpy()
    pred_masks = prediction[0]["masks"].cpu().numpy()

    # Filter predictions by threshold
    pred_boxes = pred_boxes[pred_scores >= threshold]
    pred_masks = pred_masks[pred_scores >= threshold]
    pred_scores = pred_scores[pred_scores >= threshold]

    return pred_boxes, pred_masks, pred_scores


def visualize_results(img, boxes, masks, scores, threshold=0.5):
    img_np = np.array(img)

    for i, box in enumerate(boxes):
        if scores[i] >= threshold:
            xmin, ymin, xmax, ymax = box.astype(int)
            # Draw the rectangle
            cv2.rectangle(img_np, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            # Apply the mask
            mask = masks[i, 0] > 0
            img_np[mask] = [255, 0, 0]  # Red color for mask

    return Image.fromarray(img_np)


def main():
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

    # Process images and generate predictions
    input_folder = os.path.join(args.Validation_Folder, "input_images")
    output_folder = args.output_path
    os.makedirs(output_folder, exist_ok=True)

    # Sort the files based on their numerical suffixes
    input_images = sorted(os.listdir(input_folder))

    # Add tqdm to display the progress bar
    for input_image_name in tqdm(input_images, desc="Processing images"):
        if input_image_name.endswith(".png"):
            image_path = os.path.join(input_folder, input_image_name)
            img = Image.open(image_path).convert("RGB")

            boxes, masks, scores = run_inference(
                model, device, image_path, threshold=0.5
            )
            predicted_mask = visualize_results(img, boxes, masks, scores, threshold=0.5)
            predictions_list.append(predicted_mask)

            # Save prediction mask
            predicted_mask.save(os.path.join(output_folder, f"pred_{input_image_name}"))


if __name__ == "__main__":
    main()
