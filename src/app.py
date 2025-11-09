import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
import numpy as np

# Custom Dataset Class
class CarBrandDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        """
        Args:
            root_dir: Directory with subdirectories for each car brand
            transform: Optional transform to be applied on images
            train: Boolean to indicate train/test mode
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.train = train
        
        # Get all brand folders and create label mapping
        self.brands = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.brand_to_idx = {brand: idx for idx, brand in enumerate(self.brands)}
        self.idx_to_brand = {idx: brand for brand, idx in self.brand_to_idx.items()}
        
        # Collect all image paths and labels
        self.samples = []
        for brand in self.brands:
            brand_dir = self.root_dir / brand
            for img_path in brand_dir.glob('*.[jp][pn]g'):  # .jpg, .png, .jpeg
                self.samples.append((str(img_path), self.brand_to_idx[brand]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Model Definition
class CarBrandClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(CarBrandClassifier, self).__init__()
        # Use ResNet50 as backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Replace the final fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_car_brand_model.pth')
            print(f'Saved best model with accuracy: {best_acc:.4f}')
    
    return model

# Prediction Function
def predict_image(model, image_path, transform, idx_to_brand, device):
    """Predict the car brand for a single image"""
    model.eval()
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    predicted_brand = idx_to_brand[predicted.item()]
    confidence_score = confidence.item()
    
    return predicted_brand, confidence_score

# Main execution
if __name__ == '__main__':
    # Configuration
    TRAIN_DIR = 'path/to/train'  # Change this to your training data directory
    VAL_DIR = 'path/to/val'      # Change this to your validation data directory
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = CarBrandDataset(TRAIN_DIR, transform=train_transform, train=True)
    val_dataset = CarBrandDataset(VAL_DIR, transform=val_transform, train=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Get number of classes
    num_classes = len(train_dataset.brands)
    print(f'Number of car brands: {num_classes}')
    print(f'Brands: {train_dataset.brands}')
    
    # Create model
    model = CarBrandClassifier(num_classes=num_classes, pretrained=True).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train the model
    print('\nStarting training...')
    model = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device)
    
    # Example prediction
    print('\n' + '='*50)
    print('Example Prediction:')
    test_image = 'path/to/test/image.jpg'  # Change this to test an image
    
    if os.path.exists(test_image):
        predicted_brand, confidence = predict_image(
            model, test_image, val_transform, 
            train_dataset.idx_to_brand, device
        )
        print(f'Predicted Brand: {predicted_brand}')
        print(f'Confidence: {confidence*100:.2f}%')
    else:
        print('Test image not found. Update the test_image path to make a prediction.')