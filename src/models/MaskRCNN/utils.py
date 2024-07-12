import os

import torch
import numpy as np
from model import instance_model

# Initialize optimizer and scheduler
params = [p for p in instance_model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=0.00001)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

EPOCHS = 100
best_val_loss = np.inf
train_losses, val_losses = [], []
train_loss_components_list, val_loss_components_list = [], []

# Checkpointing setup
CHECKPOINT_DIR = "checkpoint_spines"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")
LOG_PATH = os.path.join(CHECKPOINT_DIR, "training_log.log")
LOSS_PATH = os.path.join(CHECKPOINT_DIR, "losses.npz")

# Create checkpoint directory if it doesn't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def load_checkpoint():
    global best_val_loss
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        instance_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val_loss = checkpoint["best_val_loss"]
        start_epoch = checkpoint["epoch"] + 1
        if os.path.exists(LOSS_PATH):
            loaded_losses = np.load(LOSS_PATH)
            train_losses.extend(loaded_losses["train_losses"].tolist())
            val_losses.extend(loaded_losses["val_losses"].tolist())
    return start_epoch


def save_checkpoint(epoch):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": instance_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        CHECKPOINT_PATH,
    )
    np.savez(LOSS_PATH, train_losses=train_losses, val_losses=val_losses)

    formatted_train_loss_components = " | ".join(
        [f"{k}: {v:.4f}" for k, v in train_loss_components_list[-1].items()]
    )
    formatted_val_loss_components = " | ".join(
        [f"{k}: {v:.4f}" for k, v in val_loss_components_list[-1].items()]
    )

    with open(LOG_PATH, "a") as logfile:
        logfile.write(f"Epoch {epoch + 1}/{EPOCHS}\n")
        logfile.write(
            f"Train Loss: {train_losses[-1]:.4f} | {formatted_train_loss_components}\n"
        )
        logfile.write(
            f"Valid Loss: {val_losses[-1]:.4f} | {formatted_val_loss_components}\n\n"
        )
