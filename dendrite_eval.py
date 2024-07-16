import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

def get_model():
    # Load the model with the recommended weights
    model = fcn_resnet50(weights=None, progress=True)
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)
    return model

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    if transform:
        image = transform(image)
    return image

def load_mask(mask_path):
    mask = Image.open(mask_path).convert('L')  # Convert to grayscale
    if transform:
        mask = transform(mask)
    mask = np.array(mask)
    mask = (mask > 0).astype(np.uint8)  # Ensure binary mask
    return mask

def calculate_metrics(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    iou = intersection / union if union != 0 else 0
    precision = intersection / pred_mask.sum() if pred_mask.sum() != 0 else 0
    recall = intersection / true_mask.sum() if true_mask.sum() != 0 else 0
    return iou, precision, recall

# Define paths
input_folder = 'Dataset/DeepD3_Validation/input_images'
dendrite_folder = 'Dataset/DeepD3_Validation/dendrite_images'

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model().to(device)

# Load the trained model weights
model.load_state_dict(torch.load('dendrite_model.pt', map_location=device), strict=False)
model.eval()  # Set the model to evaluation mode

# Load image and mask paths
input_images = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])
dendrite_images = sorted([f for f in os.listdir(dendrite_folder) if f.endswith('.png')])


ious, precisions, recalls = [], [], []

with torch.no_grad():
    for input_image, dendrite_image in zip(input_images, dendrite_images):
        # Load and preprocess images and masks
        input_image_path = os.path.join(input_folder, input_image)
        dendrite_image_path = os.path.join(dendrite_folder, dendrite_image)
        
        image = load_image(input_image_path, transform).unsqueeze(0).to(device)
        true_mask = load_mask(dendrite_image_path)
        
        # Perform inference
        output = model(image)['out']
        pred_mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        
        # Calculate metrics
        iou, precision, recall = calculate_metrics(pred_mask, true_mask)
        ious.append(iou)
        precisions.append(precision)
        recalls.append(recall)

# Compute average metrics
mean_iou = np.mean(ious)
mean_precision = np.mean(precisions)
mean_recall = np.mean(recalls)

print(f'Mean IoU: {mean_iou:.4f}')
print(f'Mean Precision: {mean_precision:.4f}')
print(f'Mean Recall: {mean_recall:.4f}')
