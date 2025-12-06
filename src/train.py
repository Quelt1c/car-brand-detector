# File: src/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from src import model
from src.model import create_model
from src.data_preprocessing import get_data_loaders

# --- Hyperparameters (Task 2.3) ---
LEARNING_RATE = 1e-5
BATCH_SIZE = 32
NUM_EPOCHS = 10000  # Start with 10 and increase as needed
NUM_CLASSES = 196
VAL_SPLIT = 0.2
NUM_WORKERS = 4

# --- Paths ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TRAIN_CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'train_annotations.csv')
TRAIN_ROOT_DIR = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'cars_train', 'cars_train')
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'car_brand_detector_resnet50.pth')

def main():
    """
    Main function to run the training and validation loop.
    """
    # --- Setup (Task 2.2) ---
    
    # 1. Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Create DataLoaders
    print("Loading data...")
    train_loader, val_loader = get_data_loaders(
        root_dir=TRAIN_ROOT_DIR,
        annotations_file=TRAIN_CSV_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        val_split=VAL_SPLIT
    )
    print(f"Loaded {len(train_loader)} training batches and {len(val_loader)} validation batches.")

    # 3. Initialize Model
    print("Creating model...")
    model = create_model(num_classes=NUM_CLASSES, freeze_base=False)
    model.to(device)

    if os.path.exists(MODEL_SAVE_PATH):
        print(f"🔍 Saved model weights found at: {MODEL_SAVE_PATH}")
        print("📥 Loading model weights...")
        try:
            # Load the model "brains" from the file
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
            print("✅ Success! Continuing training from where we left off.")
        except Exception as e:
            print(f"❌ Error loading model (architecture might have changed?): {e}")
            print("⚠️ Starting from scratch.")
    else:
        print("🆕 No saved model found. Starting training from scratch.")

    # 4. Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    
    # We are only training the final 'fc' layer, as the base is frozen
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # --- Training Loop (Task 2.2) ---
    print("Starting training...")
    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (images, labels) in enumerate(train_loader):
            # Move data to device
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # --- Track Statistics (Task 2.3) ---
            running_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if (i + 1) % 100 == 0:
                print(f"  Batch {i + 1}/{len(train_loader)}: Train Loss: {loss.item():.4f}")

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100 * correct_train / total_train
        print(f"Epoch {epoch + 1} Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%")

        # --- Validation Phase ---
        model.eval()
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_acc = 100 * correct_val / total_val
        print(f"Epoch {epoch + 1} Validation Acc: {epoch_val_acc:.2f}%")

        # Save the model if it has the best validation accuracy so far
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            
            # Ensure the 'models' directory exists
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved to {MODEL_SAVE_PATH} (Val Acc: {best_val_acc:.2f}%)")

    print("\n--- Finished Training ---")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Final model saved at: {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    main()