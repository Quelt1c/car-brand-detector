import ast
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CarsDataset(Dataset):
    def __init__(self, root_dir, annotations_file, transform=None, is_test=False):
        """
        Args:
            root_dir (str): Directory with all the images.
            annotations_file (str): Path to the CSV file with annotations.
            transform (callable, optional): Optional transform to be applied
                on an image.
        """
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotations_file)
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")

        # Get bounding box coordinates from the dictionary string
        bbox_str = self.annotations.iloc[idx, 1]
        # Clean the bbox string to remove numpy type annotations
        bbox_str = bbox_str.replace("np.uint8(", "").replace("np.uint16(", "").replace(")", "")
        # Safely evaluate the string as a Python literal
        bbox = ast.literal_eval(bbox_str)
        x1 = bbox['x1']
        y1 = bbox['y1']
        x2 = bbox['x2']
        y2 = bbox['y2']

        # Crop image to bounding box
        image = image.crop((x1, y1, x2, y2))

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image
        else:
            label = self.annotations.iloc[idx, 2] - 1 # 'class_id' is at index 2
            return image, label

if __name__ == '__main__':
    pass
