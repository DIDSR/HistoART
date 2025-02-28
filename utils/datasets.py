from pathlib import Path
import torch
import torchvision
from torchvision.io import ImageReadMode
from torchvision import transforms
from torch.utils.data import DataLoader
import os

class CombinedClassDataset(torch.utils.data.Dataset):
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
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image      = torchvision.io.read_image(image_path, mode=ImageReadMode.RGB).float() / 255.0
        label      = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
def returnDataLoader(basepath, classes):
        basepath = Path(basepath)

        class0 = [basepath / classes[0]]
        class1 = [basepath / classes[1]]

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset    = CombinedClassDataset(class0, class1, transform)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

        return dataloader