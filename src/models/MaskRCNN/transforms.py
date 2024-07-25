import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define the transformation pipeline for training data
train_transform = A.Compose(
    [
        # Normalize the image with mean and standard deviation values used in pre-trained models
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        
        # Convert image to tensor format required by PyTorch
        ToTensorV2()
    ]
)

# Define the transformation pipeline for validation data
val_transform = A.Compose(
    [
        # Normalize the image with mean and standard deviation values used in pre-trained models
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        
        # Convert image to tensor format required by PyTorch
        ToTensorV2()
    ]
)

