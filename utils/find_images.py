import csv
import glob
import os

import numpy as np
from PIL import Image

from utils import set_environment


def find_images(directory, extensions):
    """
    Traverse the directory and find all files with given extensions.
    """
    image_files = []
    for root, _, _ in os.walk(directory):
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(root, f'*{ext}')))
    return image_files


def get_image_properties(file_path):
    """
    Get the dimensions and number of channels of the image file.
    """
    with Image.open(file_path) as img:
        width, height = img.size
        if img.mode in ['RGB', 'RGBA']:
            num_channels = len(img.getbands())
        else:
            num_channels = 1
        return width, height, num_channels


def write_to_csv(file_list, csv_filename):
    """
    Write the list of file paths, their dimensions, and number of channels to a CSV file with an ID for each file.
    """
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'File Path', 'Width', 'Height', 'Number of Channels'])
        for idx, file_path in enumerate(file_list, start=1):
            width, height, num_channels = get_image_properties(file_path)
            writer.writerow([idx, file_path, width, height, num_channels])


def process_images_in_batches(image_files, batch_size, target_size):
    """
    Process images in batches to calculate mean and std, resizing images to target_size.
    """
    mean_accum = np.zeros(3)
    std_accum = np.zeros(3)
    total_images = 0

    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        batch_images = []

        for file_path in batch_files:
            with Image.open(file_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = img.resize(target_size)  # Resize image to the target size
                np_image = np.array(img)
                batch_images.append(np_image)

        if batch_images:
            batch_images = np.stack(batch_images, axis=0)
            batch_mean = np.mean(batch_images, axis=(0, 1, 2))
            batch_std = np.std(batch_images, axis=(0, 1, 2))
            batch_size_actual = len(batch_images)

            mean_accum = (mean_accum * total_images + batch_mean * batch_size_actual) / (
                    total_images + batch_size_actual)
            std_accum = (std_accum * total_images + batch_std * batch_size_actual) / (total_images + batch_size_actual)
            total_images += batch_size_actual

    return mean_accum, std_accum


if __name__ == "__main__":
    data_prefix, results_prefix = set_environment()

    # Define the root directory and the file extensions to search for
    root_directory = os.path.join(data_prefix, "train")
    file_extensions = ['.png', '.jpeg']

    # Define the target size for resizing images
    target_size = (256, 256)  # Example size, adjust according to your needs

    # Find all image files
    images = find_images(root_directory, file_extensions)

    # Write the found image files and their properties to a CSV file with IDs
    csv_filename = os.path.join(results_prefix, 'image_files.csv')
    write_to_csv(images, csv_filename)

    # Process images in batches to calculate mean and standard deviation
    batch_size = 50  # Adjust batch size according to your system's memory capacity
    mean, std = process_images_in_batches(images, batch_size, target_size)

    print(f"Paths and properties of image files have been written to {csv_filename}")
    print(f"Mean of all images: {mean}")
    print(f"Standard deviation of all images: {std}")
