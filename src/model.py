# File: src/model.py

import torch
import torch.nn as nn
import torchvision.models as models

def create_model(num_classes=196, freeze_base=True):
    """
    Creates a model architecture based on a pre-trained ResNet50.

    Args:
        num_classes (int): The number of output classes (196 for Stanford Cars).
        freeze_base (bool): Whether to freeze the convolutional base layers
                            for feature extraction.
    """
    # Load a pre-trained ResNet50 model
    # Using the recommended modern weights API
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    if freeze_base:
        # Freeze all parameters in the base model
        # This is standard practice for transfer learning
        for param in model.parameters():
            param.requires_grad = False

    # Get the number of input features for the final layer
    num_ftrs = model.fc.in_features

    # Replace the final fully connected layer with a new one.
    # The new nn.Linear layer will have requires_grad=True by default,
    # so it will be the only part of the model that gets trained.
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

if __name__ == '__main__':
    # This block allows you to run this file directly to test the model
    # python -m src.model
    
    model = create_model(num_classes=196, freeze_base=True)
    
    print("--- Model Architecture ---")
    print(model)
    
    print("\n--- Final Classification Layer ---")
    print(model.fc)

    # Count trainable parameters (should just be the new fc layer)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable_params}")
    print(f"Total parameters:     {total_params}")

    # Test with a dummy input
    dummy_input = torch.randn(4, 3, 224, 224) # (batch_size, channels, H, W)
    output = model(dummy_input)
    print(f"\nOutput shape for a batch of 4: {output.shape}") # Should be (4, 196)