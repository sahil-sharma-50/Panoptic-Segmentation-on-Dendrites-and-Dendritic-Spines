import os
import torch
import numpy as np

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoint_spines")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")
LOSS_PATH = os.path.join(CHECKPOINT_DIR, "losses.npz")
LOG_PATH = os.path.join(CHECKPOINT_DIR, "training_log.log")


def load_checkpoint(model, optimizer):
    global best_val_loss
    best_val_loss = np.inf
    start_epoch = 0
    train_losses, val_losses = [], []
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val_loss = checkpoint["best_val_loss"]
        start_epoch = checkpoint["epoch"] + 1
        if os.path.exists(LOSS_PATH):
            loaded_losses = np.load(LOSS_PATH)
            train_losses.extend(loaded_losses["train_losses"].tolist())
            val_losses.extend(loaded_losses["val_losses"].tolist())
    return start_epoch, best_val_loss, train_losses, val_losses


def save_checkpoint(
    model,
    optimizer,
    epoch,
    best_val_loss,
    train_losses,
    val_losses,
    train_loss_components,
    val_loss_components,
    EPOCHS,
):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        CHECKPOINT_PATH,
    )
    np.savez(LOSS_PATH, train_losses=train_losses, val_losses=val_losses)

    formatted_train_loss_components = " | ".join(
        [f"{k}: {v:.4f}" for k, v in train_loss_components.items()]
    )
    formatted_val_loss_components = " | ".join(
        [f"{k}: {v:.4f}" for k, v in val_loss_components.items()]
    )

    with open(LOG_PATH, "a") as logfile:
        logfile.write(f"Epoch {epoch + 1}/{EPOCHS}\n")
        logfile.write(
            f"Train Loss: {train_losses[-1]:.4f} | {formatted_train_loss_components}\n"
        )
        logfile.write(
            f"Valid Loss: {val_losses[-1]:.4f} | {formatted_val_loss_components}\n"
        )
        logfile.write("\n")
