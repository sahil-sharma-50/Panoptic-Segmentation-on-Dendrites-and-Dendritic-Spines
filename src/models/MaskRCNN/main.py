import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import tqdm

from dataset import SpineDataset
from model import get_model_instance_segmentation
from transforms import train_transform, val_transform
from utils import collate_fn
from train import train_one_epoch, validate_one_epoch
from checkpoint import load_checkpoint, save_checkpoint

import warnings
warnings.filterwarnings("ignore")

# Set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_classes = 2  # Background and spine

# Initialize model
model = get_model_instance_segmentation(num_classes)
model.to(device)

# Load dataset and dataloaders
root_train = "Dataset/DeepD3_Training"
root_val = "Dataset/DeepD3_Validation"

train_loader = DataLoader(
    SpineDataset(root_train, transforms=train_transform),
    batch_size=1,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    SpineDataset(root_val, transforms=val_transform),
    batch_size=1,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_fn,
)

# Initialize optimizer and scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=0.00001)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

# Load checkpoint if exists
start_epoch, best_val_loss, train_losses, val_losses = load_checkpoint(model, optimizer)

# Training loop
EPOCHS = 100
train_loss_components_list, val_loss_components_list = [], []

for epoch in range(start_epoch, EPOCHS):
    # Train for one epoch
    avg_train_loss, train_loss_components = train_one_epoch(
        model, optimizer, train_loader, device, epoch, EPOCHS
    )
    
    # Validate for one epoch
    avg_val_loss, val_loss_components = validate_one_epoch(
        model, val_loader, device, epoch, EPOCHS
    )

    # Append losses and loss components
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_loss_components_list.append(train_loss_components)
    val_loss_components_list.append(val_loss_components)

    # Update learning rate scheduler
    lr_scheduler.step(avg_val_loss)

    # Format and print loss components
    formatted_train_loss_components = " | ".join(
        [f"{k}: {v:.4f}" for k, v in train_loss_components.items()]
    )
    formatted_val_loss_components = " | ".join(
        [f"{k}: {v:.4f}" for k, v in val_loss_components.items()]
    )

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {avg_train_loss:.4f} | {formatted_train_loss_components}")
    print(f"Val Loss: {avg_val_loss:.4f} | {formatted_val_loss_components}")

    # Save model checkpoint if validation loss improves
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "spines_model.pt")

    # Save detailed checkpoint with loss components
    save_checkpoint(
        model,
        optimizer,
        epoch,
        best_val_loss,
        train_losses,
        val_losses,
        train_loss_components,
        val_loss_components,
        EPOCHS,
    )

