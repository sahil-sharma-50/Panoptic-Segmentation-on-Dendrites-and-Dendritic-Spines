import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import DendritesDataset
from model import get_model
from train import train_fn, eval_fn
from transforms import train_transform, val_transform
from checkpoint import load_checkpoint, save_checkpoint

# Define constants
EPOCHS = 100
BATCH_SIZE = 4
NUM_WORKERS = 2

# Define data loaders
root_train = "Dataset/DeepD3_Training"
root_val = "Dataset/DeepD3_Validation"

train_loader = DataLoader(
    DendritesDataset(root_train, transforms=train_transform),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

val_loader = DataLoader(
    DendritesDataset(root_val, transforms=val_transform),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

# Initialize model, criterion, optimizer, and scheduler
model = get_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

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

# Load checkpoint if available
start_epoch, best_valid_loss = load_checkpoint(model, optimizer, metrics)

# Training and validation loop
for epoch in range(start_epoch, EPOCHS):
    # Training phase
    train_loss, train_accuracy, train_precision, train_recall, train_iou = train_fn(
        train_loader, model, criterion, optimizer, epoch, EPOCHS, device
    )

    # Validation phase
    valid_loss, valid_accuracy, valid_precision, valid_recall, valid_iou = eval_fn(
        val_loader, model, criterion, epoch, EPOCHS, device
    )

    # Update metrics
    metrics["train_losses"].append(train_loss)
    metrics["valid_losses"].append(valid_loss)
    metrics["train_accuracies"].append(train_accuracy)
    metrics["valid_accuracies"].append(valid_accuracy)
    metrics["train_precisions"].append(train_precision)
    metrics["valid_precisions"].append(valid_precision)
    metrics["train_recalls"].append(train_recall)
    metrics["valid_recalls"].append(valid_recall)
    metrics["train_ious"].append(train_iou)
    metrics["valid_ious"].append(valid_iou)

    # Print metrics for the current epoch
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(
        f"Train Loss: {train_loss:.4f} | Acc: {train_accuracy:.4f} | Precision: {train_precision:.4f} | Recall: {train_recall:.4f} | IoU: {train_iou:.4f}"
    )
    print(
        f"Valid Loss: {valid_loss:.4f} | Acc: {valid_accuracy:.4f} | Precision: {valid_precision:.4f} | Recall: {valid_recall:.4f} | IoU: {valid_iou:.4f}"
    )

    # Save the model checkpoint if the validation loss has improved
    if valid_loss < best_valid_loss:
        torch.save(model.state_dict(), "dendrite_model.pt")
        best_valid_loss = valid_loss

    # Save the checkpoint and metrics
    save_checkpoint(epoch, model, optimizer, best_valid_loss, metrics, EPOCHS)
    lr_scheduler.step(valid_loss)
