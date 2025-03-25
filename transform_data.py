import os
import random
import shutil
from PIL import Image
from torchvision import transforms

# Define the original dataset directory and the output directory
dataset_root = "data" 
output_root = "data_transform"  

# Delete the existing output directory if it exists
if os.path.exists(output_root):
    shutil.rmtree(output_root)

# Ensure the output directory exists
os.makedirs(output_root, exist_ok=True)

# Get a list of category subdirectories
categories = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]

# Define transformations
rotate_transform = transforms.RandomRotation(degrees=30)  # Rotate randomly within ±30 degrees
flip_transform = transforms.RandomHorizontalFlip(p=0.5)  # 50% chance to flip

# Process each category folder
for category in categories:
    category_path = os.path.join(dataset_root, category)
    output_category_path = os.path.join(output_root, category)

    # Ensure category folder exists in the output directory
    os.makedirs(output_category_path, exist_ok=True)

    # Get image files in the current category folder
    image_files = [f for f in os.listdir(category_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Determine 20% of images for rotation and flipping
    total_images = len(image_files)
    rotate_count = max(1, total_images // 5)  
    flip_count = max(1, total_images // 5)

    # Randomly select images for transformations
    rotate_images = random.sample(image_files, rotate_count)
    flip_images = random.sample(image_files, flip_count)

    # Apply transformations and overwrite in data_transform
    for image_file in image_files:
        image_path = os.path.join(category_path, image_file)
        output_image_path = os.path.join(output_category_path, image_file)
        image = Image.open(image_path)

        # Apply rotation if selected
        if image_file in rotate_images:
            image = rotate_transform(image)

        # Apply flip if selected
        if image_file in flip_images:
            image = flip_transform(image)

        # Save the (possibly transformed) image in the output folder, overwriting the original
        image.save(output_image_path)

print("Transformations applied successfully in 'data_transform'.")