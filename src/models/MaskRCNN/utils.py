def collate_fn(batch):
    """
    Custom collate function for PyTorch DataLoader.
    This function is used to combine a list of samples into a single batch.

    Parameters:
    - batch (list of tuples): Batch of data points.

    Returns:
    - tuple of lists: Collated batch.
    """
    return tuple(zip(*batch))


def compute_loss(model, images, targets):
    """
    Compute the loss for a given batch of images and targets using the model.

    Parameters:
    - model (torch.nn.Module): The model.
    - images (list or torch.Tensor): Input images.
    - targets (list or torch.Tensor): Ground truth targets.

    Returns:
    - loss_dict (dict): Components of the loss.
    - losses (torch.Tensor): Total loss.
    """
    model.train()
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    model.eval()
    return loss_dict, losses
