import torch
import numpy as np
import albumentations as A
from PIL import Image
import cv2
from albumentations.pytorch import ToTensorV2
from model import instance_model
import matplotlib.pyplot as plt


def load_model(model_path, num_classes, device):
    model = instance_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def run_inference(model, device, image_path, threshold=0.5):
    # Define the transformation pipeline for inference
    transform = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    # Load and preprocess the input image using Albumentations
    img = Image.open(image_path).convert("RGB")
    img = np.array(img)
    transformed = transform(image=img)
    img_tensor = transformed["image"]
    img_tensor = img_tensor.unsqueeze(0).to(device)

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

    return img, pred_boxes, pred_masks, pred_scores


def visualize_results(img, boxes, masks, scores, threshold=0.7):
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


def display_results(dset, idx, image_path, predicted_mask):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(Image.open(image_path))
    plt.axis("off")
    plt.title(f"{dset} Image [{idx}]")

    plt.subplot(1, 3, 2)
    image_path = f"Dataset/DeepD3_{dset}/spine_images/spine_{idx}.png"
    mask = Image.open(image_path).convert("L")
    binary_mask = mask.point(lambda p: p > 0 and 255)
    plt.imshow(binary_mask, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.title("Spines Mask")

    plt.subplot(1, 3, 3)
    plt.imshow(predicted_mask)
    plt.axis("off")
    plt.title("Predicted Mask")
    plt.show()


# Set device and number of classes
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_classes = 2  # Background and spine

# Load the model
model_path = "spines_model.pt"
model = load_model(model_path, num_classes, device)
# print('Model Loaded !!')

# Perform inference
idx = 15
dset = "Training"
image_path = f"Dataset/DeepD3_{dset}/input_images/image_{idx}.png"
img, boxes, masks, scores = run_inference(model, device, image_path, threshold=0.5)
predicted_mask = visualize_results(img, boxes, masks, scores, threshold=0.5)

# Display the results
display_results(dset, idx, image_path, predicted_mask)
