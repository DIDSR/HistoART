from   pathlib import Path
import torch
import torchvision
from   torchvision.io import ImageReadMode
from   torchvision import transforms
from   torch.utils.data import DataLoader
from   pyfeats import glcm_features, lbp_features
from   PIL import Image
import os
import numpy as np
from   tqdm import tqdm

class CombinedClassDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset that loads images from specified directories and assigns labels.

    Args:
        clean_paths (list of str): List of directories containing artifact-free images.
        artifact_paths (list of str): List of directories containing artifact images.
        transform (callable, optional): A function/transform to apply to the images.

    The dataset:
    - Recursively loads `.png` images from the given directories.
    - Assigns label `0` to artifact-free images and `1` to artifact images.
    - Supports optional transformations.
    """

    def __init__(self, clean_paths, artifact_paths, transform=None):
        self.image_paths = []
        self.labels      = []
        self.transform   = transform
        
        for clean_dir in clean_paths:
            for subdir, _, files in os.walk(clean_dir):
                for file in files:
                    if file.endswith(('.png')):
                        file_path = os.path.join(subdir, file)
                        self.image_paths.append(file_path)
                        self.labels.append(0)
        
        for artifact_dir in artifact_paths:
            for subdir, _, files in os.walk(artifact_dir):
                for file in files:
                    if file.endswith(('.png')):
                        file_path = os.path.join(subdir, file)
                        self.image_paths.append(file_path)
                        self.labels.append(1)

    def __len__(self):
        """Returns the total number of images in the dataset."""

        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding label.

        Args:
            idx (int): Index of the image.

        Returns:
            tuple: (image, label), where image is a tensor normalized to [0,1] and 
                   label is either 0 (artifact-free) or 1 (artifact).
        """

        image_path = self.image_paths[idx]
        image      = torchvision.io.read_image(image_path, mode=ImageReadMode.RGB).float() / 255.0
        label      = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
def returnDataLoader(basepath, classes, batch_size):
        """
        Creates a PyTorch DataLoader for the CombinedClassDataset.

        Args:
            basepath (str or Path): Base directory containing class subdirectories.
            classes (list of str): Names of the subdirectories representing the two classes 
                                (e.g., ['artifact_free', 'artifact']).

        Returns:
            DataLoader: A PyTorch DataLoader with batch size 16 and no shuffling.
        """
        basepath = Path(basepath)
        class0   = [basepath / classes[0]]
        class1   = [basepath / classes[1]]

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset    = CombinedClassDataset(class0, class1, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        return dataloader

def extract_features(image):
    """
    Extracts features from an image.
    
    Args:
        image (PIL.Image): Input image.
        
    Returns:
        dict: A dictionary containing extracted features.
    """
    grayscale_image = np.array(image.convert('L'))
    mask            = np.ones(grayscale_image.shape[:2], dtype=np.uint8)

    glcm_features_mean, glcm_features_range, _, _ = glcm_features(grayscale_image, ignore_zeros=False)
    lbp, _                                        = lbp_features(grayscale_image, mask, P=[8, 16, 24], R=[1, 2, 3])

    features_dict = {
        "glcm_mean": glcm_features_mean,
        "glcm_range": glcm_features_range,
        "lbp": lbp,
    }

    return features_dict

def extract_selected_features(images, labels, selected_features=['glcm_mean', 'glcm_range', 'lbp']):
    """
    Extracts and concatenates selected handcrafted features from a list of images.

    Args:
        images (list of PIL.Image.Image): List of input images.
        labels (list or np.ndarray): Corresponding class labels for each image.
        selected_features (list of str, optional): List of feature names to extract from each image.
                                                   Default includes ['glcm_mean', 'glcm_range', 'lbp'].

    Returns:
        tuple:
            - X (np.ndarray): Array of shape (n_samples, n_features) containing concatenated feature vectors.
            - y (np.ndarray): Array of shape (n_samples,) containing the corresponding labels.

    Raises:
        KeyError: If a selected feature is not found in the extracted feature dictionary for an image.
    """

    X = []
    y = []
    
    for img, label in tqdm(zip(images, labels), total=len(images), desc="Extracting features"):
        feats           = extract_features(img)
        feature_vectors = []
        
        for feat_name in selected_features:
            if feat_name not in feats:
                raise KeyError(f"Feature '{feat_name}' not found in extracted features.")
            feature_vectors.append(np.ravel(feats[feat_name]))
            
        concatenated = np.concatenate(feature_vectors)
        
        X.append(concatenated)
        y.append(label)
        
    return np.array(X), np.array(y)

def load_images_and_labels(root_dir, class_names):
    """
    Loads images and their corresponding labels from specified class directories.

    Args:
        root_dir (str): Root directory containing subdirectories for each class.
        class_names (list of str): List of class subdirectory names. The index of each 
                                   class name in the list is used as the label.

    Returns:
        tuple: 
            - images (list of PIL.Image.Image): List of loaded RGB images.
            - labels (np.ndarray): Numpy array of integer labels corresponding to each image.
    """
    images = []
    labels = []
    
    for cname in class_names:
        cdir = os.path.join(root_dir, cname)
        if not os.path.isdir(cdir):
            continue
        img_files = [f for f in os.listdir(cdir) if f.lower().endswith('.png')]

        label_value = class_names.index(cname)
        
        for img_file in tqdm(img_files, desc=f"Loading '{cname}'"):
            img_path = os.path.join(cdir, img_file)
            image    = Image.open(img_path).convert('RGB')
            images.append(image)
            labels.append(label_value)
    
    return images, np.array(labels)
