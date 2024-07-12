import torch
from model import instance_model
import tqdm
from transforms import train_transform, val_transform
from dataset import SpineDataset
from torch.utils.data import DataLoader
from utils import *


# Collate function for DataLoader
def collate_fn(batch):
    return tuple(zip(*batch))


# Dataset and DataLoader
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


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_classes = 2
model = instance_model(num_classes)
model.to(device)


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


def compute_loss(model, images, targets):
    model.train()
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    model.eval()
    return loss_dict, losses


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


# Load checkpoint if exists
start_epoch = load_checkpoint()

# Training loop
for epoch in range(start_epoch, EPOCHS):
    avg_train_loss, train_loss_components = train_one_epoch(
        model, optimizer, train_loader, device, epoch, EPOCHS
    )
    avg_val_loss, val_loss_components = validate_one_epoch(
        model, val_loader, device, epoch, EPOCHS
    )

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_loss_components_list.append(train_loss_components)
    val_loss_components_list.append(val_loss_components)

    lr_scheduler.step(avg_val_loss)

    formatted_train_loss_components = " | ".join(
        [f"{k}: {v:.4f}" for k, v in train_loss_components.items()]
    )
    formatted_val_loss_components = " | ".join(
        [f"{k}: {v:.4f}" for k, v in val_loss_components.items()]
    )

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {avg_train_loss:.4f} | {formatted_train_loss_components}")
    print(f"Val Loss: {avg_val_loss:.4f} | {formatted_val_loss_components}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "spines_model.pt")

    save_checkpoint(epoch)
