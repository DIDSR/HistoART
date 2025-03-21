import os
import random
import matplotlib.pyplot as plt
from PIL import Image

def visualize_images_from_classes(data_dir, classes, num_samples=5):
    """
    Displays a grid of randomly sampled images from specified class folders.

    Args:
        data_dir (str): Root directory containing subdirectories for each class.
        classes (list of str): List of class subdirectory names to visualize.
        num_samples (int, optional): Number of images to display per class. Default is 5.

    The function:
    - Iterates over the provided class names.
    - Randomly samples up to `num_samples` images from each class folder.
    - Displays the sampled images in a grid with one row per class.
    - Handles missing folders or empty folders with informative messages.

    Notes:
    - Supported image extensions include .png, .jpg, .jpeg, .bmp, and .gif.
    - If a class folder doesn't exist or contains no images, a message is shown in its place.
    """
    
    num_classes = len(classes)
    if num_classes == 0:
        print("No classes provided.")
        return
    
    fig, axes = plt.subplots(num_classes, num_samples, figsize=(5 * num_samples, 5 * num_classes))
    if num_classes == 1:
        axes = [axes]
    
    for row, cls in enumerate(classes):
        class_folder = os.path.join(data_dir, cls)
        if not os.path.isdir(class_folder):
            for col in range(num_samples):
                axes[row][col].set_title(cls if col == 0 else "")
                axes[row][col].text(0.5, 0.5, "Folder not found", horizontalalignment='center', verticalalignment='center')
                axes[row][col].axis("off")
            continue
        
        image_files = [f for f in os.listdir(class_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        if not image_files:
            for col in range(num_samples):
                axes[row][col].set_title(cls if col == 0 else "")
                axes[row][col].text(0.5, 0.5, "No images found", horizontalalignment='center', verticalalignment='center')
                axes[row][col].axis("off")
            continue
        
        sampled_images = random.sample(image_files, min(num_samples, len(image_files)))
        for col, image_file in enumerate(sampled_images):
            image_path = os.path.join(class_folder, image_file)
            img = Image.open(image_path)
            axes[row][col].imshow(img)
            axes[row][col].axis("off")
            if col == 0:
                axes[row][col].set_title(cls)
    
    plt.tight_layout()
    plt.show()
