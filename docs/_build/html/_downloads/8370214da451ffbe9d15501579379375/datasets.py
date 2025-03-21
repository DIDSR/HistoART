import torch
import torchvision
import os
import concurrent.futures
import numpy as np
from   pathlib          import Path
from   torchvision.io   import ImageReadMode
from   torchvision      import transforms
from   torch.utils.data import DataLoader
from   pyfeats          import glcm_features, lbp_features
from   PIL              import Image
from   tqdm             import tqdm


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
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

        return dataloader

class FeatureLoadingDataset(torch.utils.data.Dataset):
    def __init__(self, features_root):
        """
        Loads pre-extracted feature files from disk.
        
        Args:
            features_root (str): Root directory where features are saved.
        """
        self.samples = []

        for class_name in os.listdir(features_root):
            class_dir = os.path.join(features_root, class_name)

            if not os.path.isdir(class_dir):
                continue

            for feature_type in os.listdir(class_dir):
                feature_dir = os.path.join(class_dir, feature_type)

                for file in os.listdir(feature_dir):
                    
                    if file.endswith('.npy'):
                        file_path = os.path.join(feature_dir, file)
                        self.samples.append(file_path)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path = self.samples[idx]
        data = np.load(file_path, allow_pickle=True).item()
        feature = data["feature"]
        label = data["label"]
        feature_tensor = torch.tensor(feature, dtype=torch.float32).view(-1)
        return feature_tensor, label

def returnFeatureLoaderFromDisk(features_root, batch_size=64, num_workers=8):
    """
    Creates a PyTorch DataLoader for the FeatureLoadingDataset.
    
    Args:
        features_root (str): Root directory where features are saved.
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes for DataLoader.
        
    Returns:
        DataLoader: A DataLoader loading pre-extracted features.
    """
    dataset    = FeatureLoadingDataset(features_root)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader

def convert_dataloader_to_numpy(dataloader):
    """
    Converts a DataLoader (yielding feature tensors and labels)
    to NumPy arrays suitable for SVM prediction.
    """
    features_list = []
    labels_list = []
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

def process_image(image_path, class_name, output_root):
    """
    Processes a single image: opens it, extracts features, and saves each feature.
    
    Args:
        image_path (str): Path to the image file.
        class_name (str): Class label of the image.
        output_root (str): Directory to save the extracted features.
    """
    try:
        image = Image.open(image_path)

    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return
    
    features_dict = extract_features(image)
    image_id = os.path.splitext(os.path.basename(image_path))[0]

    for key, feature in features_dict.items():
        feature_output_dir = os.path.join(output_root, class_name, key)

        os.makedirs(feature_output_dir, exist_ok=True)
        save_path = os.path.join(feature_output_dir, image_id + ".npy")
        np.save(save_path, {"feature": feature, "label": class_name})

    image.close()

def process_and_save_features_parallel(input_root, output_root, max_workers=8):
    """
    Processes images in parallel from subdirectories and saves their features in separate folders.
    
    Args:
        input_root (str): Root directory containing class subdirectories with images.
        output_root (str): Root directory to save extracted features.
        max_workers (int): Number of worker processes.
    """
    for class_name in os.listdir(input_root):
        class_input_dir = os.path.join(input_root, class_name)

        if not os.path.isdir(class_input_dir):
            continue

        image_files = [f for f in os.listdir(class_input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_paths = [os.path.join(class_input_dir, f) for f in image_files]

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_image, image_path, class_name, output_root) for image_path in image_paths]

            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing {class_name}"):
                pass
