import os
import sys
import pandas as pd
from scipy.io import loadmat
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Adjust path to import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset import CarsDataset

def load_annotations(mat_file_path):
    """Loads annotations from a .mat file."""
    return loadmat(mat_file_path)

def load_class_names(meta_file_path):
    """Loads class names from a .mat file."""
    return [name[0] for name in loadmat(meta_file_path)['class_names'][0]]

def process_annotations(annotations, class_names):
    """Processes raw training annotations."""
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
        class_name = class_names[class_id - 1]
        processed_data.append({
            'fname': fname,
            'bbox': str(bbox),  # Store bbox as a string
            'class_id': class_id,
            'class_name': class_name
        })
    return processed_data

def process_test_annotations(annotations):
    """Processes raw test annotations (without class labels)."""
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
        processed_data.append({
            'fname': fname,
            'bbox': str(bbox), # Store bbox as a string
        })
    return processed_data

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def get_data_loaders(root_dir, annotations_file, batch_size=32, num_workers=4, val_split=0.2):
    """Creates and returns training and validation data loaders."""
    full_dataset = CarsDataset(root_dir=root_dir, annotations_file=annotations_file, transform=data_transforms['train'])
    train_size = int((1 - val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = data_transforms['val']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader

def get_test_loader(root_dir, annotations_file, batch_size=32, num_workers=4):
    """Creates and returns a test data loader."""
    test_dataset = CarsDataset(root_dir=root_dir, annotations_file=annotations_file, transform=data_transforms['test'], is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_loader

if __name__ == '__main__':
    # Define project root and construct relative paths
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    devkit_path = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'car_devkit', 'devkit')
    train_annos_path = os.path.join(devkit_path, 'cars_train_annos.mat')
    test_annos_path = os.path.join(devkit_path, 'cars_test_annos.mat')
    meta_path = os.path.join(devkit_path, 'cars_meta.mat')

    # --- Process Training Data ---
    class_names = load_class_names(meta_path)
    train_mat_data = load_annotations(train_annos_path)
    processed_train_data = process_annotations(train_mat_data['annotations'], class_names)
    train_df = pd.DataFrame(processed_train_data)
    train_csv_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'train_annotations.csv')
    train_df.to_csv(train_csv_path, index=False)
    print(f"Processed {len(train_df)} training annotations and saved to {train_csv_path}")

    # --- Process Test Data ---
    test_mat_data = load_annotations(test_annos_path)
    processed_test_data = process_test_annotations(test_mat_data['annotations'])
    test_df = pd.DataFrame(processed_test_data)
    test_csv_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'test_annotations.csv')
    test_df.to_csv(test_csv_path, index=False)
    print(f"Processed {len(test_df)} test annotations and saved to {test_csv_path}")

    # --- Get and Test Data Loaders ---
    train_root_dir = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'cars_train', 'cars_train')
    test_root_dir = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'cars_test', 'cars_test')

    train_loader, val_loader = get_data_loaders(
        root_dir=train_root_dir,
        annotations_file=train_csv_path,
        batch_size=4,
        num_workers=2
    )
    print(f"\nCreated {len(train_loader)} training batches and {len(val_loader)} validation batches.")

    test_loader = get_test_loader(
        root_dir=test_root_dir,
        annotations_file=test_csv_path,
        batch_size=4,
        num_workers=2
    )
    print(f"Created {len(test_loader)} test batches.")

    # --- Verify Loaders ---
    print("\nVerifying data loaders...")
    train_images, train_labels = next(iter(train_loader))
    print(f"Train batch images shape: {train_images.shape}")
    print(f"Train batch labels: {train_labels}")

    val_images, val_labels = next(iter(val_loader))
    print(f"Validation batch images shape: {val_images.shape}")
    print(f"Validation batch labels: {val_labels}")

    test_images = next(iter(test_loader))
    print(f"Test batch images shape: {test_images.shape}")
