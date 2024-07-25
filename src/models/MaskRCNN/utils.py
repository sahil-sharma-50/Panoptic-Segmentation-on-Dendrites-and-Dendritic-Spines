import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score

def calculate_metrics(outputs: torch.Tensor, masks: torch.Tensor):
    """
    Calculates various performance metrics for binary segmentation.

    Parameters:
    - outputs (torch.Tensor): The raw outputs from the model (logits).
    - masks (torch.Tensor): The ground truth masks.

    Returns:
    - Tuple[float, float, float, float]: A tuple containing accuracy, precision, recall, and IoU.
    """
    # Apply sigmoid activation to the outputs to get probabilities
    outputs = torch.sigmoid(outputs)
    
    # Convert probabilities to binary predictions (threshold at 0.5)
    outputs = (outputs > 0.5).float()

    # Move tensors to CPU, detach from the computation graph, convert to numpy arrays, 
    # and flatten them for metric calculations
    outputs = outputs.cpu().detach().numpy().astype(np.uint8).flatten()
    masks = masks.cpu().detach().numpy().astype(np.uint8).flatten()

    # Calculate accuracy using sklearn's accuracy_score function
    accuracy = accuracy_score(masks, outputs)
    
    # Calculate precision; zero_division=1 prevents division by zero errors
    precision = precision_score(masks, outputs, zero_division=1)
    
    # Calculate recall; zero_division=1 prevents division by zero errors
    recall = recall_score(masks, outputs, zero_division=1)

    # Calculate the Intersection over Union (IoU)
    intersection = np.sum((masks * outputs) > 0)
    union = np.sum((masks + outputs) > 0)
    iou = intersection / union if union > 0 else 0.0

    # Return the calculated metrics
    return accuracy, precision, recall, iou
