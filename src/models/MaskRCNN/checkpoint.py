import os
import torch
import numpy as np

# Define directory paths for saving and loading checkpoints and logs
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoint_spines")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")
LOSS_PATH = os.path.join(CHECKPOINT_DIR, "losses.npz")
LOG_PATH = os.path.join(CHECKPOINT_DIR, "training_log.log")


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
    """
    Saves the model and optimizer states, and logs the losses and loss components.
    
    Parameters:
    - model (torch.nn.Module): The model to save state_dict from.
    - optimizer (torch.optim.Optimizer): The optimizer to save state_dict from.
    - epoch (int): The current epoch number.
    - best_val_loss (float): The best validation loss recorded.
    - train_losses (list): List of training losses to save.
    - val_losses (list): List of validation losses to save.
    - train_loss_components (dict): Dictionary of training loss components to log.
    - val_loss_components (dict): Dictionary of validation loss components to log.
    - EPOCHS (int): Total number of epochs for training.
    """
    # Create checkpoint directory if it does not exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Save model and optimizer states
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        CHECKPOINT_PATH,
    )
    
    # Save training and validation losses
    np.savez(LOSS_PATH, train_losses=train_losses, val_losses=val_losses)

    # Format loss components for logging
    formatted_train_loss_components = " | ".join(
        [f"{k}: {v:.4f}" for k, v in train_loss_components.items()]
    )
    formatted_val_loss_components = " | ".join(
        [f"{k}: {v:.4f}" for k, v in val_loss_components.items()]
    )

    # Append training log
    with open(LOG_PATH, "a") as logfile:
        logfile.write(f"Epoch {epoch + 1}/{EPOCHS}\n")
        logfile.write(
            f"Train Loss: {train_losses[-1]:.4f} | {formatted_train_loss_components}\n"
        )
        logfile.write(
            f"Valid Loss: {val_losses[-1]:.4f} | {formatted_val_loss_components}\n"
        )
        logfile.write("\n")


def load_checkpoint(model, optimizer):
    """
    Loads the model and optimizer states from the checkpoint if available.
    
    Parameters:
    - model (torch.nn.Module): The model to load state_dict into.
    - optimizer (torch.optim.Optimizer): The optimizer to load state_dict into.
    
    Returns:
    - start_epoch (int): The epoch to resume training from.
    - best_val_loss (float): The best validation loss recorded.
    - train_losses (list): List of training losses loaded from the checkpoint.
    - val_losses (list): List of validation losses loaded from the checkpoint.
    """
    global best_val_loss
    best_val_loss = np.inf
    start_epoch = 0
    train_losses, val_losses = [], []

    # Check if checkpoint file exists
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val_loss = checkpoint["best_val_loss"]
        start_epoch = checkpoint["epoch"] + 1

        # Check if losses file exists and load losses
        if os.path.exists(LOSS_PATH):
            loaded_losses = np.load(LOSS_PATH)
            train_losses.extend(loaded_losses["train_losses"].tolist())
            val_losses.extend(loaded_losses["val_losses"].tolist())

    return start_epoch, best_val_loss, train_losses, val_losses

