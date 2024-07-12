import torch.nn as nn
from torchvision.models.segmentation import FCN_ResNet50_Weights
from torchvision.models.segmentation import fcn_resnet50


def semantic_model(num_classes):
    weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    model = fcn_resnet50(weights=weights, progress=True)
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)
    return model
