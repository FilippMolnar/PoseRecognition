import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import imagehash
from PIL import Image

def dhash(image, hash_size=8):
    """Compute the difference hash (dHash) of an image."""
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def find_duplicate_images(directory, threshold=5):
    """Find and remove duplicate images in a given directory."""
    image_hashes = defaultdict(list)
    files = list(Path(directory).glob("*.jpg")) + \
            list(Path(directory).glob("*.png")) + \
            list(Path(directory).glob("*.jpeg"))

    # Compute hashes
    for file in tqdm(files, desc="Scanning images"):
        try:
            image = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            hash_value = dhash(image)
            image_hashes[hash_value].append(file)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    # Identify duplicates
    duplicates = []
    for hash_val, paths in image_hashes.items():
        if len(paths) > 1:
            duplicates.extend(paths[1:])  # Keep the first one, delete others

    # Remove duplicate images
    for duplicate in tqdm(duplicates, desc="Removing duplicates"):
        try:
            os.remove(duplicate)
            print(f"Deleted: {duplicate}")
        except Exception as e:
            print(f"Error deleting {duplicate}: {e}")

if __name__ == "__main__":
    directory = "data/stag_leap"
    find_duplicate_images(directory)
