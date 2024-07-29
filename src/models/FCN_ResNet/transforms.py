import torchvision.transforms as transforms

# Transformation pipeline for training images
train_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),  # Resize the image to 256x256 pixels
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    ]
)

# Transformation pipeline for validation images
val_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),  # Resize the image to 256x256 pixels
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    ]
)
