import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score

def calculate_metrics(outputs, masks):
    outputs = torch.sigmoid(outputs)
    outputs = (outputs > 0.5).float()
    
    outputs = outputs.cpu().detach().numpy().astype(np.uint8).flatten()
    masks = masks.cpu().detach().numpy().astype(np.uint8).flatten()
    
    accuracy = accuracy_score(masks, outputs)
    precision = precision_score(masks, outputs, zero_division=1)
    recall = recall_score(masks, outputs, zero_division=1)
    
    intersection = np.sum((masks * outputs) > 0)
    union = np.sum((masks + outputs) > 0)
    iou = intersection / union if union > 0 else 0.0
    
    return accuracy, precision, recall, iou
