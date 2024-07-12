import os
import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DendriteDataset
from model import semantic_model
from transforms import train_transform, val_transform
from utils import *

root_train = "Dataset/DeepD3_Training"
root_val = "Dataset/DeepD3_Validation"

train_loader = DataLoader(
    DendriteDataset(root_train, transforms=train_transform),
    batch_size=4,
    shuffle=True,
    num_workers=2,
)

val_loader = DataLoader(
    DendriteDataset(root_val, transforms=val_transform),
    batch_size=4,
    shuffle=False,
    num_workers=2,
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_classes = 2
model = semantic_model(num_classes)
model.to(device)


def train_fn(data_loader, model, criterion, optimizer, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    total_accuracy = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_iou = 0.0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")

    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device).float()

        optimizer.zero_grad()
        outputs = model(images)["out"]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        accuracy, precision, recall, iou = calculate_metrics(outputs, masks)
        total_accuracy += accuracy * images.size(0)
        total_precision += precision * images.size(0)
        total_recall += recall * images.size(0)
        total_iou += iou * images.size(0)

        progress_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_accuracy = total_accuracy / len(data_loader.dataset)
    epoch_precision = total_precision / len(data_loader.dataset)
    epoch_recall = total_recall / len(data_loader.dataset)
    epoch_iou = total_iou / len(data_loader.dataset)

    return epoch_loss, epoch_accuracy, epoch_precision, epoch_recall, epoch_iou


def eval_fn(data_loader, model, criterion, epoch, num_epochs):
    model.eval()
    running_loss = 0.0
    total_accuracy = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_iou = 0.0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")
    with torch.no_grad():
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device).float()

            outputs = model(images)["out"]
            loss = criterion(outputs, masks)

            running_loss += loss.item() * images.size(0)
            accuracy, precision, recall, iou = calculate_metrics(outputs, masks)
            total_accuracy += accuracy * images.size(0)
            total_precision += precision * images.size(0)
            total_recall += recall * images.size(0)
            total_iou += iou * images.size(0)
            progress_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_accuracy = total_accuracy / len(data_loader.dataset)
    epoch_precision = total_precision / len(data_loader.dataset)
    epoch_recall = total_recall / len(data_loader.dataset)
    epoch_iou = total_iou / len(data_loader.dataset)

    return epoch_loss, epoch_accuracy, epoch_precision, epoch_recall, epoch_iou


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)


start_epoch = load_checkpoint()
for epoch in range(start_epoch, EPOCHS):
    train_loss, train_accuracy, train_precision, train_recall, train_iou = train_fn(
        train_loader, model, criterion, optimizer, epoch, EPOCHS
    )
    valid_loss, valid_accuracy, valid_precision, valid_recall, valid_iou = eval_fn(
        val_loader, model, criterion, epoch, EPOCHS
    )

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

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(
        f"Train Loss: {train_loss:.4f} | Acc: {train_accuracy:.4f} | Precision: {train_precision:.4f} | Recall: {train_recall:.4f} | IoU: {train_iou:.4f}"
    )
    print(
        f"Valid Loss: {valid_loss:.4f} | Acc: {valid_accuracy:.4f} | Precision: {valid_precision:.4f} | Recall: {valid_recall:.4f} | IoU: {valid_iou:.4f}"
    )

    if valid_loss < best_valid_loss:
        torch.save(model.state_dict(), "dendrite_model.pt")
        best_valid_loss = valid_loss

    save_checkpoint(epoch)
    lr_scheduler.step(valid_loss)
