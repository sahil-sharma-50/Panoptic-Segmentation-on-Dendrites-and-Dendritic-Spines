import torch
import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

def get_model():
    # Load the model with the recommended weights
    weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    model = fcn_resnet50(weights=weights, progress=True)
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)
    return model
