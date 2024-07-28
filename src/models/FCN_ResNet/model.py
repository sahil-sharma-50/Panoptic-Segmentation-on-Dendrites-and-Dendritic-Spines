import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

def get_model():
    """
    Initializes and returns a modified FCN-ResNet50 model for binary segmentation.

    The model is pre-trained on the COCO dataset with VOC labels and is modified 
    to output a single channel for binary segmentation tasks.

    Returns:
        model (nn.Module): The modified FCN-ResNet50 model.
    """
    # Load the FCN-ResNet50 model pre-trained on COCO with VOC labels
    weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    model = fcn_resnet50(weights=weights, progress=True)

    # Modify the classifier to output a single channel (for binary segmentation)
    # (input channels, output channels, kernel size)
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)

    return model

