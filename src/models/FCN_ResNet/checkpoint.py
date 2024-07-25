import os
import torch
import numpy as np

# Define directory paths for checkpoints and metrics
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoint_dendrites")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "model_checkpoint.pth")
METRICS_PATH = os.path.join(CHECKPOINT_DIR, "metrics.npz")
LOG_PATH = os.path.join(CHECKPOINT_DIR, "metrics_log.log")

# Create the checkpoint directory if it doesn't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def load_checkpoint(model, optimizer, metrics):
    """
    Load the model and optimizer state from a checkpoint if it exists.

    Parameters:
    - model: The model to load the state into.
    - optimizer: The optimizer to load the state into.
    - metrics: A dictionary to load saved metrics.

    Returns:
    - start_epoch (int): The epoch to start training from.
    - best_valid_loss (float): The best validation loss seen so far.
    """
    best_valid_loss = np.Inf  # Initialize best validation loss to infinity
    start_epoch = 0  # Initialize start epoch

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_valid_loss = checkpoint["best_valid_loss"]
        start_epoch = checkpoint["epoch"] + 1  # Start from the next epoch

        if os.path.exists(METRICS_PATH):
            loaded_metrics = np.load(METRICS_PATH)
            for key in metrics:
                metrics[key] = loaded_metrics[key].tolist()
                
    return start_epoch, best_valid_loss

def save_checkpoint(epoch, model, optimizer, best_valid_loss, metrics, EPOCHS):
    """
    Save the model and optimizer state to a checkpoint and log metrics.

    Parameters:
    - epoch (int): The current epoch number.
    - model: The model to save.
    - optimizer: The optimizer to save.
    - best_valid_loss (float): The best validation loss seen so far.
    - metrics (dict): A dictionary of training and validation metrics.
    - EPOCHS (int): The total number of epochs.
    """
    # Save model and optimizer state to checkpoint
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_valid_loss": best_valid_loss,
        },
        CHECKPOINT_PATH,
    )
    # Save metrics to a .npz file
    np.savez(METRICS_PATH, **metrics)

    # Log metrics to a log file
    with open(LOG_PATH, "a") as logfile:
        logfile.write(f"Epoch {epoch + 1}/{EPOCHS}\n")
        logfile.write(
            f"Train Loss: {metrics['train_losses'][-1]:.4f} | Acc: {metrics['train_accuracies'][-1]:.4f} | Precision: {metrics['train_precisions'][-1]:.4f} | Recall: {metrics['train_recalls'][-1]:.4f} | IoU: {metrics['train_ious'][-1]:.4f}\n"
        )
        logfile.write(
            f"Valid Loss: {metrics['valid_losses'][-1]:.4f} | Acc: {metrics['valid_accuracies'][-1]:.4f} | Precision: {metrics['valid_precisions'][-1]:.4f} | Recall: {metrics['valid_recalls'][-1]:.4f} | IoU: {metrics['valid_ious'][-1]:.4f}\n\n"
        )

