import matplotlib.pyplot as plt
from PIL import Image
import random
import os

def visualize_images_from_classes(data_dir, classes):
    num_classes = len(classes)
    if num_classes == 0:
        print("No classes provided.")
        return
    fig, axes = plt.subplots(1, num_classes, figsize=(5 * num_classes, 5))
    if num_classes == 1:
        axes = [axes]
    for ax, cls in zip(axes, classes):
        class_folder = os.path.join(data_dir, cls)
        if not os.path.isdir(class_folder):
            ax.set_title(cls)
            ax.text(0.5, 0.5, "Folder not found", horizontalalignment='center', verticalalignment='center')
            ax.axis("off")
            continue
        image_files = [f for f in os.listdir(class_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        if not image_files:
            ax.set_title(cls)
            ax.text(0.5, 0.5, "No images found", horizontalalignment='center', verticalalignment='center')
            ax.axis("off")
            continue
        image_path = os.path.join(class_folder, random.choice(image_files))
        img = Image.open(image_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(cls)
    plt.tight_layout()
    plt.show()