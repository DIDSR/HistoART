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
    
class FeatureExtractionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, selected_features=['glcm_mean', 'glcm_range', 'lbp'],
                 transform=None, save_features=False, output_root=None):
        """
        Args:
            root_dir (str): Directory with subfolders for each class.
            selected_features (list): List of feature keys to use and save.
            transform (callable, optional): Transform to apply to the extracted feature vector.
            save_features (bool): If True, save each extracted feature separately.
            output_root (str): Root directory where features will be saved.
        """
        self.root_dir          = root_dir
        self.selected_features = selected_features
        self.transform         = transform
        self.save_features     = save_features
        self.output_root       = output_root
        self.samples           = []
        
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for file in os.listdir(class_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_dir, file)
                    self.samples.append((image_path, class_name))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, class_name = self.samples[idx]

        try:
            image = Image.open(image_path)
        except Exception as e:
            raise RuntimeError(f"Error opening image {image_path}: {e}")
        
        features_dict  = extract_features(image)
        feature_vector = []

        for key in self.selected_features:
            if key in features_dict and features_dict[key] is not None:
                feature_vector.append(np.ravel(features_dict[key]))

        if feature_vector:
            feature_vector = np.concatenate(feature_vector)
        else:
            feature_vector = np.array([])

        if self.save_features and self.output_root is not None:
            image_id = os.path.splitext(os.path.basename(image_path))[0]

            for key in self.selected_features:
                feature = features_dict.get(key)

                if feature is not None:
                    feature_out_dir = os.path.join(self.output_root, class_name, key)
                    os.makedirs(feature_out_dir, exist_ok=True)
                    save_path = os.path.join(feature_out_dir, image_id + ".npy")
                    np.save(save_path, {"feature": feature, "label": class_name})

        image.close()
        feature_tensor = torch.tensor(feature_vector, dtype=torch.float32)
        if self.transform:
            feature_tensor = self.transform(feature_tensor)
        return feature_tensor, class_name
    
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

def returnFeatureLoader(basepath, selected_features=['glcm_mean', 'glcm_range', 'lbp'], transform=None, save_features=True, output_root="./features"):
    """
    Creates a PyTorch DataLoader for the FeatureExtractionDataset.

    Args:
        basepath (str or Path): Root directory containing subdirectories for each class.
        selected_features (list): List of feature keys to extract and use.
        transform (callable, optional): Transform to apply to the extracted feature vector.
        save_features (bool): If True, saves each extracted feature separately.
        output_root (str or Path, optional): Root directory where features will be saved if save_features is True.

    Returns:
        DataLoader: A PyTorch DataLoader with batch size 16 and no shuffling.
    """
    dataset = FeatureExtractionDataset(root_dir=basepath, selected_features=selected_features, transform=transform, save_features=save_features, output_root=output_root)
    dataloader    = DataLoader(dataset, batch_size=16, shuffle=False)

    return dataloader


def convert_dataloader_to_numpy(dataloader):
    """
    Converts a DataLoader (yielding feature tensors and labels)
    to NumPy arrays suitable for SVM prediction.
    """
    features_list = []
    labels_list   = []
    
    for inputs, labels in tqdm(dataloader, desc="Converting DataLoader to NumPy"):

        inputs = inputs.cpu().view(inputs.size(0), -1)
        features_list.append(inputs.numpy())

        if hasattr(labels, "cpu"):
            labels_list.append(labels.cpu().numpy())
        else:
            labels_list.append(np.array(labels))

    X = np.concatenate(features_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    return X, y

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

def process_and_save_features(input_root, output_root):
    """
    Processes images organized in subfolders (by class) and saves their features
    in separate folders for each class and feature type.
    
    Args:
        input_root (str): Root directory containing class subdirectories with images.
        output_root (str): Root directory to save extracted features.
    """

    for class_name in os.listdir(input_root):
        class_input_dir = os.path.join(input_root, class_name)
        if not os.path.isdir(class_input_dir):
            continue

        for feature_key in ['glcm_mean', 'glcm_range', 'lbp']:
            feature_output_dir = os.path.join(output_root, class_name, feature_key)
            os.makedirs(feature_output_dir, exist_ok=True)

        for image_file in tqdm(os.listdir(class_input_dir), desc=f"Processing {class_name}"):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(class_input_dir, image_file)
                try:
                    image = Image.open(image_path)
                except Exception as e:
                    print(f"Error opening image {image_path}: {e}")
                    continue

                features_dict = extract_features(image)
                image_id      = os.path.splitext(image_file)[0]

                for key, feature in features_dict.items():

                    save_path = os.path.join(output_root, class_name, key, image_id + ".npy")
                    np.save(save_path, {"feature": feature, "label": class_name})
                    
                image.close()
