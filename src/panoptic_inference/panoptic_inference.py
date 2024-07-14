import numpy as np
import torch
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import functional as F
import argparse
import os
from tqdm import tqdm
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.segmentation import fcn_resnet50
import torchvision.models as models
import random


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_model_instance_segmentation(num_classes):
    weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
    model = models.detection.maskrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 512
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model


def load_models(instance_model_path, semantic_model_path, device):
    instance_model = get_model_instance_segmentation(num_classes=2)
    instance_model.load_state_dict(
        torch.load(instance_model_path, map_location=device), strict=False
    )
    instance_model.to(device)
    instance_model.eval()

    semantic_model = fcn_resnet50(weights=None, num_classes=1)
    semantic_model.load_state_dict(
        torch.load(semantic_model_path, map_location=device), strict=False
    )
    semantic_model.to(device)
    semantic_model.eval()

    return instance_model, semantic_model


def run_instance_inference(model, device, image_path, threshold=0.5):
    transform = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    img = Image.open(image_path).convert("RGB")
    img = np.array(img)
    transformed = transform(image=img)
    img_tensor = transformed["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(img_tensor)
    pred_boxes = prediction[0]["boxes"].cpu().numpy()
    pred_scores = prediction[0]["scores"].cpu().numpy()
    pred_masks = prediction[0]["masks"].cpu().numpy()
    pred_boxes = pred_boxes[pred_scores >= threshold]
    pred_masks = pred_masks[pred_scores >= threshold]
    pred_scores = pred_scores[pred_scores >= threshold]
    return img, pred_boxes, pred_masks, pred_scores


def run_semantic_inference(model, device, image_path):
    input_image = Image.open(image_path)
    image = F.to_tensor(input_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)["out"]
    return input_image, (torch.sigmoid(output).squeeze().cpu().numpy() > 0.1).astype(
        np.uint8
    )


def random_color():
    return [random.randint(0, 255) for _ in range(3)]


def create_panoptic_mask_with_colors(
    instance_masks, semantic_mask, scores, threshold=0.3
):
    height, width = semantic_mask.shape
    panoptic_mask = np.zeros((height, width, 3), dtype=np.uint8)
    panoptic_mask[semantic_mask == 1] = (0, 255, 0)

    for mask, score in zip(instance_masks, scores):
        if score > threshold:
            mask_bin = mask.squeeze() > threshold
            unique_color = random_color()
            panoptic_mask[mask_bin] = unique_color
            contours, _ = cv2.findContours(
                mask_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(panoptic_mask, contours, -1, (255, 0, 0), 1)
    return panoptic_mask


def main(instance_model_path, semantic_model_path, input_images_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    device = get_device()
    instance_model, semantic_model = load_models(
        instance_model_path, semantic_model_path, device
    )

    image_files = sorted(
        [f for f in os.listdir(input_images_folder) if f.endswith(".png")]
    )

    for idx, image_name in enumerate(tqdm(image_files, desc="Processing images")):
        if image_name.endswith(".png"):
            image_path = os.path.join(input_images_folder, image_name)

            img, boxes, instance_masks, scores = run_instance_inference(
                instance_model, device, image_path, threshold=0.5
            )
            _, semantic_mask = run_semantic_inference(
                semantic_model, device, image_path
            )
            panoptic_mask = create_panoptic_mask_with_colors(
                instance_masks, semantic_mask, scores, threshold=0.3
            )

            output_filename = f"panoptic_{image_name}"
            output_path = os.path.join(output_folder, output_filename)
            Image.fromarray(panoptic_mask).save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Panoptic Segmentation Script")
    parser.add_argument(
        "--instance_model_path",
        type=str,
        required=True,
        help="Path to the instance model.",
    )
    parser.add_argument(
        "--semantic_model_path",
        type=str,
        required=True,
        help="Path to the semantic model.",
    )
    parser.add_argument(
        "--input_images_folder",
        type=str,
        required=True,
        help="Path to the input images folder.",
    )
    parser.add_argument(
        "--output_folder", type=str, required=True, help="Path to the output folder."
    )
    args = parser.parse_args()
    main(
        args.instance_model_path,
        args.semantic_model_path,
        args.input_images_folder,
        args.output_folder,
    )
