import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from .dataset import CarsDataset

def load_annotations(mat_file_path):
    """
    Loads annotations from a .mat file.
    Args:
        mat_file_path (str): Path to the .mat annotation file.
    Returns:
        dict: Loaded annotations.
    """
    return loadmat(mat_file_path)

def load_class_names(meta_file_path):
    """
    Loads class names from a .mat file.
    Args:
        meta_file_path (str): Path to the .mat meta file.
    Returns:
        list: List of class names.
    """
    return [name[0] for name in loadmat(meta_file_path)['class_names'][0]]

def process_annotations(annotations, class_names):
    """
    Processes raw annotations into a more usable format.
    Args:
        annotations (dict): Raw annotations loaded from .mat file.
        class_names (list): List of class names.
    Returns:
        list: A list of dictionaries, each containing 'fname', 'bbox', 'class_id', and 'class_name'.
    """
    processed_data = []
    for i in range(annotations.shape[1]):
        annotation = annotations[0, i]
        fname = annotation['fname'][0]
        bbox = {
            'x1': annotation['bbox_x1'][0, 0],
            'y1': annotation['bbox_y1'][0, 0],
            'x2': annotation['bbox_x2'][0, 0],
            'y2': annotation['bbox_y2'][0, 0],
        }
        class_id = annotation['class'][0, 0]
        class_name = class_names[class_id - 1]  # Adjust for 0-based indexing

        processed_data.append({
            'fname': fname,
            'bbox': bbox,
            'class_id': class_id,
            'class_name': class_name
        })
    return processed_data

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def get_data_loaders(root_dir, annotations_file, batch_size=32, num_workers=4, val_split=0.2):
    """
    Creates and returns training and validation data loaders.
    Args:
        root_dir (str): Directory with all the images.
        annotations_file (str): Path to the CSV file with annotations.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        val_split (float): Fraction of the dataset to use for validation.
    Returns:
        tuple: (train_loader, val_loader)
    """
    full_dataset = CarsDataset(root_dir=root_dir, annotations_file=annotations_file, transform=data_transforms['train'])

    # Split dataset into training and validation
    train_size = int((1 - val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Apply validation transforms to the validation dataset
    # Note: This requires re-creating the dataset or a custom way to apply different transforms
    # For simplicity, we'll assume the transform is applied during __getitem__ and can be changed.
    # A more robust solution would involve creating two separate dataset instances with different transforms.
    val_dataset.dataset.transform = data_transforms['val']

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    devkit_path = '/home/sonorma/Documents/uni/car-brand-detector/data/dataset/car_devkit/devkit'
    train_annos_path = os.path.join(devkit_path, 'cars_train_annos.mat')
    meta_path = os.path.join(devkit_path, 'cars_meta.mat')

    # Load class names
    class_names = load_class_names(meta_path)
    print(f"Loaded {len(class_names)} class names.")

    # Load and process training annotations
    train_mat_data = load_annotations(train_annos_path)
    train_annotations = train_mat_data['annotations']
    processed_train_data = process_annotations(train_annotations, class_names)
    print(f"Processed {len(processed_train_data)} training annotations.")

    df_train = pd.DataFrame(processed_train_data)
    processed_train_csv_path = 'data/processed/train_annotations.csv'
    df_train.to_csv(processed_train_csv_path, index=False)
    print(f"Processed training data saved to {processed_train_csv_path}")

    # Get data loaders
    train_root_dir = '/home/sonorma/Documents/uni/car-brand-detector/data/dataset/cars_train/cars_train'
    train_loader, val_loader = get_data_loaders(
        root_dir=train_root_dir,
        annotations_file=processed_train_csv_path,
        batch_size=4,
        num_workers=2
    )

    print(f"Number of batches in training loader: {len(train_loader)}")
    print(f"Number of batches in validation loader: {len(val_loader)}")

    # Test a batch from the training loader
    for images, labels in train_loader:
        print(f"Train Batch image tensor shape: {images.shape}")
        print(f"Train Batch labels: {labels}")
        break

    # Test a batch from the validation loader
    for images, labels in val_loader:
        print(f"Val Batch image tensor shape: {images.shape}")
        print(f"Val Batch labels: {labels}")
        break