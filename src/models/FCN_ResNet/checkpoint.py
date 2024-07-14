import os
import torch
import numpy as np

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoint_spines')
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'model_checkpoint.pth')
METRICS_PATH = os.path.join(CHECKPOINT_DIR, 'metrics.npz')
LOG_PATH = os.path.join(CHECKPOINT_DIR, 'metrics_log.log')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def load_checkpoint(model, optimizer):
    best_valid_loss = np.Inf
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_valid_loss = checkpoint['best_valid_loss']
        start_epoch = checkpoint['epoch'] + 1
        if os.path.exists(METRICS_PATH):
            loaded_metrics = np.load(METRICS_PATH)
            for key in metrics:
                metrics[key] = loaded_metrics[key].tolist()
    return start_epoch, best_valid_loss

def save_checkpoint(epoch, model, optimizer, best_valid_loss, metrics, EPOCHS):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_valid_loss': best_valid_loss
    }, CHECKPOINT_PATH)
    np.savez(METRICS_PATH, **metrics)
    
    # Log metrics to log file
    with open(LOG_PATH, 'a') as logfile:
        logfile.write(f"Epoch {epoch + 1}/{EPOCHS}\n")
        logfile.write(f"Train Loss: {metrics['train_losses'][-1]:.4f} | Acc: {metrics['train_accuracies'][-1]:.4f} | Precision: {metrics['train_precisions'][-1]:.4f} | Recall: {metrics['train_recalls'][-1]:.4f} | IoU: {metrics['train_ious'][-1]:.4f}\n")
        logfile.write(f"Valid Loss: {metrics['valid_losses'][-1]:.4f} | Acc: {metrics['valid_accuracies'][-1]:.4f} | Precision: {metrics['valid_precisions'][-1]:.4f} | Recall: {metrics['valid_recalls'][-1]:.4f} | IoU: {metrics['valid_ious'][-1]:.4f}\n\n")
