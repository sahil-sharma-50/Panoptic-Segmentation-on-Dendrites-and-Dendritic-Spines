def collate_fn(batch):
    return tuple(zip(*batch))

def compute_loss(model, images, targets):
    model.train()
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    model.eval()
    return loss_dict, losses
