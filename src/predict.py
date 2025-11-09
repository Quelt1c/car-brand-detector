"""
predict.py - Single Image Prediction Script
Use this to predict car brand on new images
"""

import torch
from PIL import Image
import sys
from pathlib import Path
import argparse

from dataset import get_predict_transforms
from model import load_model


class CarBrandPredictor:
    """Handle predictions on car images"""
    
    def __init__(self, model_path, brands, device='cpu', model_type='resnet50'):
        """
        Args:
            model_path: Path to trained model (.pth file)
            brands: List of brand names in correct order
            device: 'cpu' or 'cuda'
            model_type: 'resnet50' or 'mobilenet'
        """
        self.device = torch.device(device)
        self.brands = brands
        self.idx_to_brand = {idx: brand for idx, brand in enumerate(brands)}
        self.transform = get_predict_transforms()
        
        # Load model
        num_classes = len(brands)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Import here to avoid circular dependency
        from model import create_model
        self.model = create_model(model_type, num_classes, pretrained=False)
        
        # Load state dict (handle both full checkpoint and state_dict only)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded from {model_path}")
        print(f"✓ Brands: {brands}")
    
    def predict(self, image_path, top_k=3):
        """
        Predict car brand for a single image
        
        Args:
            image_path: Path to image file
            top_k: Return top K predictions
        
        Returns:
            predictions: List of (brand, confidence) tuples
        """
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
        
        # Transform and add batch dimension
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        # Get top K predictions
        top_probs, top_indices = torch.topk(probabilities, min(top_k, len(self.brands)))
        
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            brand = self.idx_to_brand[idx.item()]
            confidence = prob.item()
            predictions.append((brand, confidence))
        
        return predictions
    
    def predict_batch(self, image_paths):
        """Predict on multiple images"""
        results = {}
        for img_path in image_paths:
            predictions = self.predict(img_path)
            results[img_path] = predictions
        return results


def main():
    """Main prediction function"""
    
    parser = argparse.ArgumentParser(description='Predict car brand from image')
    parser.add_argument('--image', type=str, required=True, 
                       help='Path to image file')
    parser.add_argument('--model', type=str, default='../models/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--brands', nargs='+', default=['AUDI', 'BMW', 'TOYOTA'],
                       help='List of brand names (in training order)')
    parser.add_argument('--model_type', type=str, default='resnet50',
                       choices=['resnet50', 'mobilenet'],
                       help='Model architecture type')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for inference')
    parser.add_argument('--top_k', type=int, default=3,
                       help='Show top K predictions')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f'Using device: {device}')
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"ERROR: Model not found at {args.model}")
        print("Please train the model first using train.py")
        return
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"ERROR: Image not found at {args.image}")
        return
    
    # Create predictor
    predictor = CarBrandPredictor(
        model_path=args.model,
        brands=args.brands,
        device=device,
        model_type=args.model_type
    )
    
    # Make prediction
    print(f'\nPredicting on: {args.image}')
    print('-' * 60)
    
    predictions = predictor.predict(args.image, top_k=args.top_k)
    
    if predictions:
        print(f'\nTop {len(predictions)} Predictions:')
        for i, (brand, confidence) in enumerate(predictions, 1):
            print(f'{i}. {brand:15s} - {confidence*100:5.2f}%')
        
        # Show top prediction
        top_brand, top_conf = predictions[0]
        print(f'\n{"="*60}')
        print(f'Predicted Brand: {top_brand}')
        print(f'Confidence: {top_conf*100:.2f}%')
        print(f'{"="*60}')
    else:
        print("Prediction failed!")


# Alternative: Simple function for quick predictions
def quick_predict(image_path, model_path='../models/best_model.pth', 
                 brands=['AUDI', 'BMW', 'TOYOTA']):
    """
    Simple function to quickly predict on an image
    
    Usage:
        from predict import quick_predict
        brand, confidence = quick_predict('my_car.jpg')
        print(f"This is a {brand} with {confidence*100:.1f}% confidence")
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = CarBrandPredictor(model_path, brands, device)
    predictions = predictor.predict(image_path, top_k=1)
    
    if predictions:
        return predictions[0]  # Return (brand, confidence)
    return None, 0.0


if __name__ == '__main__':
    # If no arguments provided, show usage example
    if len(sys.argv) == 1:
        print("Usage: python predict.py --image path/to/car.jpg")
        print("\nExample:")
        print("  python predict.py --image ../data/test/my_car.jpg")
        print("  python predict.py --image ../data/test/my_car.jpg --brands AUDI BMW TOYOTA FORD")
        print("\nFull options:")
        print("  --image      : Path to image (required)")
        print("  --model      : Path to trained model (default: ../models/best_model.pth)")
        print("  --brands     : Brand names in order (default: AUDI BMW TOYOTA)")
        print("  --model_type : resnet50 or mobilenet (default: resnet50)")
        print("  --device     : auto, cpu, or cuda (default: auto)")
        print("  --top_k      : Number of top predictions to show (default: 3)")
    else:
        main()