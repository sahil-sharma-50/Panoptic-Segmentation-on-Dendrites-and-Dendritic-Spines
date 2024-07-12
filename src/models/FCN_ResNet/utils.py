import torch
import numpy as np
import os
from dotenv import load_dotenv
import torch.nn as nn
import torch.optim as optim
from model import semantic_model
from sklearn.metrics import accuracy_score, precision_score, recall_score

load_dotenv()

EPOCHS = 100
CHECKPOINT_DIR = os.getenv("DENDERITES_CHECKPOINT_DIR")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "model_checkpoint.pth")
METRICS_PATH = os.path.join(CHECKPOINT_DIR, "metrics.npz")
LOG_PATH = os.path.join(CHECKPOINT_DIR, "metrics_log.log")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# Initialize metrics dictionary
metrics = {
    "train_losses": [],
    "valid_losses": [],
    "train_accuracies": [],
    "valid_accuracies": [],
    "train_precisions": [],
    "valid_precisions": [],
    "train_recalls": [],
    "valid_recalls": [],
    "train_ious": [],
    "valid_ious": [],
}

# Initialize criterion, optimizer, and lr_scheduler
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(semantic_model.parameters(), lr=0.0001)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)


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


def load_checkpoint():
    global best_valid_loss
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        semantic_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_valid_loss = checkpoint["best_valid_loss"]
        start_epoch = checkpoint["epoch"] + 1
        if os.path.exists(METRICS_PATH):
            loaded_metrics = np.load(METRICS_PATH)
            for key in metrics:
                metrics[key] = loaded_metrics[key].tolist()
    else:
        start_epoch = 0
    return start_epoch


def save_checkpoint(epoch):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": semantic_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_valid_loss": best_valid_loss,
        },
        CHECKPOINT_PATH,
    )
    np.savez(METRICS_PATH, **metrics)

    # Log metrics to log file
    with open(LOG_PATH, "a") as logfile:
        logfile.write(f"Epoch {epoch + 1}/{EPOCHS}\n")
        logfile.write(
            f"Train Loss: {metrics['train_losses'][-1]:.4f} | Acc: {metrics['train_accuracies'][-1]:.4f} | Precision: {metrics['train_precisions'][-1]:.4f} | Recall: {metrics['train_recalls'][-1]:.4f} | IoU: {metrics['train_ious'][-1]:.4f}\n"
        )
        logfile.write(
            f"Valid Loss: {metrics['valid_losses'][-1]:.4f} | Acc: {metrics['valid_accuracies'][-1]:.4f} | Precision: {metrics['valid_precisions'][-1]:.4f} | Recall: {metrics['valid_recalls'][-1]:.4f} | IoU: {metrics['valid_ious'][-1]:.4f}\n\n"
        )
