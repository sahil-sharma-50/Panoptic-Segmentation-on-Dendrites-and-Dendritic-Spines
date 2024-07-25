import torch
import tqdm
from utils import collate_fn, compute_loss

def train_one_epoch(model, optimizer, data_loader, device, epoch, num_epochs):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for model parameters.
        data_loader (torch.utils.data.DataLoader): DataLoader for training data.
        device (torch.device): The device to train on (CPU or GPU).
        epoch (int): The current epoch number.
        num_epochs (int): The total number of epochs for training.

    Returns:
        float: The average training loss for the epoch.
        dict: A dictionary of average loss components for the epoch.
    """
    model.train()  # Set model to training mode
    train_epoch_loss = 0
    loss_components = {"loss_box_reg": 0, "loss_classifier": 0, "loss_mask": 0}
    
    # Create a progress bar for tracking training progress
    progress_bar = tqdm.tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")

    for images, targets in progress_bar:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass and compute losses
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()  # Zero out gradients before backward pass
        losses.backward()  # Backward pass to compute gradients
        optimizer.step()  # Update model parameters

        # Accumulate losses
        train_epoch_loss += losses.item()
        for k in loss_components.keys():
            if k in loss_dict:
                loss_components[k] += loss_dict[k].item()

        # Update progress bar with current loss
        progress_bar.set_postfix(loss=losses.item())

    num_batches = len(data_loader)
    avg_loss_components = {k: v / num_batches for k, v in loss_components.items()}
    return train_epoch_loss / num_batches, avg_loss_components

def validate_one_epoch(model, data_loader, device, epoch, num_epochs):
    """
    Validates the model for one epoch.

    Args:
        model (torch.nn.Module): The model to be validated.
        data_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        device (torch.device): The device to validate on (CPU or GPU).
        epoch (int): The current epoch number.
        num_epochs (int): The total number of epochs for validation.

    Returns:
        float: The average validation loss for the epoch.
        dict: A dictionary of average loss components for the epoch.
    """
    model.eval()  # Set model to evaluation mode
    val_epoch_loss = 0
    loss_components = {"loss_box_reg": 0, "loss_classifier": 0, "loss_mask": 0}
    
    # Create a progress bar for tracking validation progress
    val_progress_bar = tqdm.tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")

    with torch.no_grad():  # Disable gradient computation
        for images, targets in val_progress_bar:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Compute losses
            loss_dict, losses = compute_loss(model, images, targets)

            # Accumulate losses
            val_epoch_loss += losses.item()
            for k in loss_components.keys():
                if k in loss_dict:
                    loss_components[k] += loss_dict[k].item()

            # Update progress bar with current loss
            val_progress_bar.set_postfix(loss=losses.item())

    num_batches = len(data_loader)
    avg_loss_components = {k: v / num_batches for k, v in loss_components.items()}
    return val_epoch_loss / num_batches, avg_loss_components

