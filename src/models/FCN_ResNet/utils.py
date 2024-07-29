import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score


def calculate_metrics(outputs, masks):
    """
    Calculate various evaluation metrics for binary segmentation.

    Parameters:
    - outputs (torch.Tensor): Model predictions. Expected to be logits or raw scores.
    - masks (torch.Tensor): Ground truth binary masks.

    Returns:
    - accuracy (float): Accuracy of the predictions.
    - precision (float): Precision of the predictions.
    - recall (float): Recall of the predictions.
    - iou (float): Intersection over Union (IoU) metric.
    """

    # Apply sigmoid activation to outputs to obtain probabilities
    outputs = torch.sigmoid(outputs)

    # Convert probabilities to binary predictions
    outputs = (outputs > 0.5).float()

    # Move tensors to CPU and convert to numpy arrays
    outputs = outputs.cpu().detach().numpy().astype(np.uint8).flatten()
    masks = masks.cpu().detach().numpy().astype(np.uint8).flatten()

    # Compute evaluation metrics
    accuracy = accuracy_score(masks, outputs)
    precision = precision_score(masks, outputs, zero_division=1)
    recall = recall_score(masks, outputs, zero_division=1)

    # Compute Intersection over Union (IoU)
    intersection = np.sum((masks * outputs) > 0)
    union = np.sum((masks + outputs) > 0)
    iou = intersection / union if union > 0 else 0.0

    return accuracy, precision, recall, iou
