import os
from PIL import Image
import numpy as np

# Define the path to the directory containing PNG images
image_dir = '/home/r948e/E132-Projekte/Projects/2024_MICCAI_Mario_Challenge/data/Task1/train'
output_dir = '/home/r948e/dev/Mario/data/Task1/npy/train/'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate through the directory, load images, and convert them to tensors
for filename in os.listdir(image_dir):
    if filename.endswith('.png'):
        print(filename)
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path)
        img_array = np.array(image)
        npy_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.npy")
        np.save(npy_path, img_array)

print(f'Converted and saved tensors for images in {image_dir}.')
