import torch
import tqdm
from utils import collate_fn, compute_loss


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_epochs):
    model.train()
    train_epoch_loss = 0
    loss_components = {"loss_box_reg": 0, "loss_classifier": 0, "loss_mask": 0}
    progress_bar = tqdm.tqdm(
        data_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"
    )

    for images, targets in progress_bar:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        train_epoch_loss += losses.item()
        for k in loss_components.keys():
            if k in loss_dict:
                loss_components[k] += loss_dict[k].item()

        progress_bar.set_postfix(loss=losses.item())

    num_batches = len(data_loader)
    avg_loss_components = {k: v / num_batches for k, v in loss_components.items()}
    return train_epoch_loss / num_batches, avg_loss_components


def validate_one_epoch(model, data_loader, device, epoch, num_epochs):
    model.eval()
    val_epoch_loss = 0
    loss_components = {"loss_box_reg": 0, "loss_classifier": 0, "loss_mask": 0}
    val_progress_bar = tqdm.tqdm(
        data_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"
    )

    with torch.no_grad():
        for images, targets in val_progress_bar:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict, losses = compute_loss(model, images, targets)

            val_epoch_loss += losses.item()
            for k in loss_components.keys():
                if k in loss_dict:
                    loss_components[k] += loss_dict[k].item()

            val_progress_bar.set_postfix(loss=losses.item())

    num_batches = len(data_loader)
    avg_loss_components = {k: v / num_batches for k, v in loss_components.items()}
    return val_epoch_loss / num_batches, avg_loss_components
