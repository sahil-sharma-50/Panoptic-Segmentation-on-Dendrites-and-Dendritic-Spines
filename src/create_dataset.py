import os
import numpy as np
import zipfile
import argparse
from tqdm import tqdm
from typing import Dict
from PIL import Image, ImageSequence
from scipy.ndimage import label
import matplotlib.pyplot as plt

# Define a type alias for counters
CountersType = Dict[str, int]

# Global counters to keep track of the number of images processed
counters: CountersType = {"image": 0, "dendrite": 0, "spine": 0}


def process_tiff(
    tiff_path: str,
    input_images_path: str,
    dendrites_images_path: str,
    spines_images_path: str,
) -> None:
    """
    Process a single TIFF image and save its pages as separate PNG images.

    Parameters:
    - tiff_path (str): Path to the TIFF file.
    - input_images_path (str): Directory to save input images.
    - dendrites_images_path (str): Directory to save dendrite images.
    - spines_images_path (str): Directory to save spine images.
    """
    tif = Image.open(tiff_path)
    for page_number, page in enumerate(ImageSequence.Iterator(tif)):
        image_array = np.array(page)
        if page_number % 3 == 0:
            save_image(image_array, input_images_path, "image")
        elif page_number % 3 == 1:
            save_image(image_array, dendrites_images_path, "dendrite")
        elif page_number % 3 == 2:
            save_spine_instance_mask(image_array, spines_images_path)


def save_image(image_array: np.ndarray, folder_path: str, prefix: str) -> None:
    """
    Save an image array as a PNG file and update the corresponding counter.

    Parameters:
    - image_array (np.ndarray): Image data.
    - folder_path (str): Directory to save the image.
    - prefix (str): Prefix for the image filename.
    """
    global counters
    image_path = os.path.join(folder_path, f"{prefix}_{counters[prefix]}.png")
    save_with_imsave(image_array, image_path)
    counters[prefix] += 1


def save_spine_instance_mask(binary_image: np.ndarray, spines_images_path: str) -> None:
    """
    Save a binary spine mask as a labeled PNG image.

    Parameters:
    - binary_image (np.ndarray): Binary mask image.
    - spines_images_path (str): Directory to save the spine images.
    """
    global counters
    labeled_image, num_features = label(binary_image)
    output_path = os.path.join(spines_images_path, f'spine_{counters["spine"]}.png')
    save_with_imsave(labeled_image.astype(np.uint8), output_path)
    counters["spine"] += 1


def save_with_imsave(image_array: np.ndarray, output_path: str) -> None:
    """
    Save an image using plt.imsave.

    Parameters:
    - image_array (np.ndarray): Image data.
    - output_path (str): Path to save the image.
    """
    plt.imsave(output_path, image_array, cmap="gray")


def process_folder(input_folder: str, output_folder: str) -> None:
    """
    Process all TIFF images in a folder.

    Parameters:
    - input_folder (str): Directory containing the TIFF images.
    - output_folder (str): Directory to save the processed images.
    """
    global counters
    counters = {"image": 0, "dendrite": 0, "spine": 0}

    input_images_path = os.path.join(output_folder, "input_images")
    dendrites_images_path = os.path.join(output_folder, "dendrite_images")
    spines_images_path = os.path.join(output_folder, "spine_images")

    # Create directories if they don't exist
    for path in [input_images_path, dendrites_images_path, spines_images_path]:
        os.makedirs(path, exist_ok=True)

    # Get a list of all TIFF files in the input folder
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


def main(zip_path: str, extract_path: str, output_path: str) -> None:
    """
    Extract a zip file and process the contained TIFF images.

    Parameters:
    - zip_path (str): Path to the zip file.
    - extract_path (str): Path to extract the zip file.
    - output_path (str): Path to save the processed images.
    """
    # Extract the zip file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        file_list = zip_ref.namelist()
        progress_bar = tqdm(file_list, desc=f"Extracting {zip_path}", unit=" files")
        for file in progress_bar:
            zip_ref.extract(file, extract_path)

    # Process each extracted folder
    for folder_name in os.listdir(extract_path):
        folder_path = os.path.join(extract_path, folder_name)
        if os.path.isdir(folder_path):
            output_folder = os.path.join(output_path, folder_name)
            process_folder(folder_path, output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a zip file containing TIFF images."
    )
    parser.add_argument(
        "--zip_path",
        type=str,
        required=True,
        help="Path to the zip file containing the images.",
    )
    parser.add_argument(
        "--extract_path",
        type=str,
        required=True,
        help="Path where the zip file will be extracted.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Folder where the processed images will be stored.",
    )

    args = parser.parse_args()
    main(args.zip_path, args.extract_path, args.output_path)
    print("Processing complete.")
