import os
from typing import Dict
import numpy as np
from PIL import Image, ImageSequence
from scipy.ndimage import label
import matplotlib.pyplot as plt
import zipfile
import argparse
from tqdm import tqdm

# Define a type alias for counters
CountersType = Dict[str, int]

# Global counters
counters: CountersType = {"image": 0, "dendrite": 0, "spine": 0}


# Function to process a single TIFF image
def process_tiff(
    tiff_path: str,
    input_images_path: str,
    dendrites_images_path: str,
    spines_images_path: str,
) -> None:
    tif = Image.open(tiff_path)
    for page_number, page in enumerate(ImageSequence.Iterator(tif)):
        image_array = np.array(page)
        if page_number % 3 == 0:
            save_image(image_array, input_images_path, "image")
        elif page_number % 3 == 1:
            save_image(image_array, dendrites_images_path, "dendrite")
        elif page_number % 3 == 2:
            save_spine_instance_mask(image_array, spines_images_path)


# Function to save an image and update the corresponding counter
def save_image(image_array: np.ndarray, folder_path: str, prefix: str) -> None:
    global counters
    image_path = os.path.join(folder_path, f"{prefix}_{counters[prefix]}.png")
    save_with_imsave(image_array, image_path)
    counters[prefix] += 1


# Function to save spine instance mask directly
def save_spine_instance_mask(binary_image: np.ndarray, spines_images_path: str) -> None:
    global counters
    labeled_image, num_features = label(binary_image)
    output_path = os.path.join(spines_images_path, f'spine_{counters["spine"]}.png')
    save_with_imsave(labeled_image.astype(np.uint8), output_path)
    counters["spine"] += 1


# Function to save images using plt.imsave
def save_with_imsave(image_array: np.ndarray, output_path: str) -> None:
    plt.imsave(output_path, image_array, cmap="gray")


# Function to process all TIFF images in a folder
def process_folder(input_folder: str, output_folder: str) -> None:
    global counters
    counters = {"image": 0, "dendrite": 0, "spine": 0}

    input_images_path = os.path.join(output_folder, "input_images")
    dendrites_images_path = os.path.join(output_folder, "dendrite_images")
    spines_images_path = os.path.join(output_folder, "spine_images")

    for path in [input_images_path, dendrites_images_path, spines_images_path]:
        os.makedirs(path, exist_ok=True)

    tiff_files = [
        file
        for file in os.listdir(input_folder)
        if file.lower().endswith((".tif", ".tiff"))
    ]
    for tiff_file in tqdm(tiff_files, desc=f"Processing TIFF files in {input_folder}"):
        process_tiff(
            os.path.join(input_folder, tiff_file),
            input_images_path,
            dendrites_images_path,
            spines_images_path,
        )


# Function to extract zip file and process images
def main(zip_path: str, extract_path: str, output_path: str) -> None:
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    for folder_name in os.listdir(extract_path):
        folder_path = os.path.join(extract_path, folder_name)
        if os.path.isdir(folder_path):
            output_folder = os.path.join(output_path, folder_name)
            print(f"Processing folder: {folder_name}")
            process_folder(folder_path, output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a zip file containing TIFF images."
    )
    parser.add_argument(
        "zip_path", type=str, help="Path to the zip file containing the images."
    )
    parser.add_argument(
        "extract_path", type=str, help="Path where the zip file will be extracted."
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Folder where the processed images will be stored.",
    )

    args = parser.parse_args()

    main(args.zip_path, args.extract_path, args.output_path)

    print("Processing complete.")
