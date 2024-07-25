import torch
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

def get_model_instance_segmentation(num_classes):
    """
    Initializes and returns a Mask R-CNN model with a specified number of classes.
    
    Args:
        num_classes (int): The number of classes for the model, including background.
        
    Returns:
        torchvision.models.detection.MaskRCNN: The modified Mask R-CNN model with the given number of classes.
    """
    # Load a pre-trained Mask R-CNN model with ResNet50-FPN backbone
    weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
    
    # Modify the box predictor to match the number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Modify the mask predictor to match the number of classes
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 512  # Number of hidden units in mask predictor
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    
    return model

