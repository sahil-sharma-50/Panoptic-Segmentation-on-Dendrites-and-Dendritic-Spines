import os
import argparse
import torch
from tqdm import tqdm
from torchvision.transforms import functional as F
from torchvision.models.segmentation import fcn_resnet50
from PIL import Image
import numpy as np


def main():
    # Define command line arguments
    parser = argparse.ArgumentParser(
        description="Inference for Dendrite Semantic Segmentation"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model"
    )
    parser.add_argument(
        "--validation_folder",
        type=str,
        required=True,
        help="Path to the validation folder",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output images and metrics",
    )
    args = parser.parse_args()

    # Load and prepare the model
    num_classes = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dendrites = fcn_resnet50(weights=None, num_classes=num_classes)
    model_dendrites.load_state_dict(torch.load(args.model_path), strict=False)
    model_dendrites.eval().to(device)

    # Process images and generate predictions
    input_folder = os.path.join(args.validation_folder, "input_images")
    output_folder = args.output_path

    os.makedirs(output_folder, exist_ok=True)

    # Sort the files based on their numerical suffixes
    input_images = sorted([f for f in os.listdir(input_folder) if f.endswith(".png")])

    # Add tqdm to display the progress bar
    for input_image_name in tqdm(input_images, desc="Processing images"):
        image_path = os.path.join(input_folder, input_image_name)

        input_image = Image.open(image_path)

        # Convert image to tensor and add batch dimension
        image = F.to_tensor(input_image.convert("RGB")).unsqueeze(0).to(device)

        # Perform Inference
        with torch.no_grad():
            output = model_dendrites(image)["out"]
        prediction = (torch.sigmoid(output).squeeze().cpu().numpy() > 0.5).astype(
            np.uint8
        )

        # Save prediction mask
        # Image formats often require pixel values in the range [0, 255]
        output_image = Image.fromarray(prediction * 255)
        output_image.save(os.path.join(output_folder, f"pred_{input_image_name}"))


if __name__ == "__main__":
    main()
