"""
dataset.py - Custom Dataset Class and Data Transforms
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path


class CarBrandDataset(Dataset):
    """Custom Dataset for Car Brand Classification"""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Directory with subdirectories for each car brand
                     Example: data/train/BMW/, data/train/AUDI/, etc.
            transform: Optional transform to be applied on images
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Get all brand folders and create label mapping
        self.brands = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.brand_to_idx = {brand: idx for idx, brand in enumerate(self.brands)}
        self.idx_to_brand = {idx: brand for brand, idx in self.brand_to_idx.items()}
        
        # Collect all image paths and labels
        self.samples = []
        self._load_samples()
        
        print(f"Loaded {len(self.samples)} images from {len(self.brands)} brands")
        print(f"Brands: {self.brands}")
    
    def _load_samples(self):
        """Load all image paths and their corresponding labels"""
        for brand in self.brands:
            brand_dir = self.root_dir / brand
            # Support common image formats
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
            for ext in image_extensions:
                for img_path in brand_dir.glob(ext):
                    self.samples.append((str(img_path), self.brand_to_idx[brand]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_brand_name(self, idx):
        """Get brand name from class index"""
        return self.idx_to_brand[idx]


def get_train_transforms():
    """
    Data augmentation transforms for training
    Includes random flips, rotations, and color adjustments
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms():
    """
    Standard transforms for validation/testing
    No augmentation, just resize and normalize
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def get_predict_transforms():
    """
    Transforms for prediction on single images
    Same as validation transforms
    """
    return get_val_transforms()