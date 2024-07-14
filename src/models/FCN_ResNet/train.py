import torch
from tqdm import tqdm
from utils import calculate_metrics

def train_fn(data_loader, model, criterion, optimizer, epoch, num_epochs, device):
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
        outputs = model(images)['out']
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

def eval_fn(data_loader, model, criterion, epoch, num_epochs, device):
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

            outputs = model(images)['out']
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
