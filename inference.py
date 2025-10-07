import torch
import argparse
from pathlib import Path
from PIL import Image
import sys
import os

# Fix for PyTorch 2.6+
torch.serialization.add_safe_globals([argparse.Namespace])

from timm.models import create_model
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import model
import utils


def load_model_and_labels(checkpoint_path, model_name='repvit_m0_9'):
    """
    Load trained model and label mapping
    
    Returns:
        model, idx_to_label, num_classes, device
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Get model state
    if 'model' not in checkpoint:
        print("ERROR: 'model' key not found in checkpoint!")
        print(f"Available keys: {checkpoint.keys()}")
        sys.exit(1)
    
    model_state = checkpoint['model']
    
    # Find number of classes by inspecting model state dict
    num_classes = None
    print("\nSearching for output layer to determine number of classes...")
    
    # List all keys that might be the output layer
    for key in model_state.keys():
        if 'weight' in key and any(x in key for x in ['head', 'fc', 'classifier']):
            weight_shape = model_state[key].shape
            print(f"  Found: {key} -> shape: {weight_shape}")
            if len(weight_shape) == 2:  # Should be [num_classes, features]
                num_classes = weight_shape[0]
                print(f"  → Using this as output layer")
                break
    
    if num_classes is None:
        print("\nERROR: Could not automatically determine number of classes.")
        print("Model state dict keys:")
        for k in sorted(model_state.keys()):
            if 'head' in k or 'fc' in k or 'classifier' in k:
                print(f"  {k}: {model_state[k].shape}")
        sys.exit(1)
    
    print(f"\n✓ Number of classes: {num_classes}")
    
    # Create model
    print(f"Creating model: {model_name}")
    net = create_model(model_name, num_classes=num_classes, pretrained=False)
    
    # Load weights
    net.load_state_dict(model_state)
    utils.replace_batchnorm(net)
    net.to(device)
    net.eval()
    print("✓ Model loaded successfully")
    
    # Load label mapping
    idx_to_label = load_label_mapping(num_classes)
    
    return net, idx_to_label, num_classes, device


def load_label_mapping(num_classes):
    """Load label names from CSV or create generic labels"""
    print("\nLoading label mapping...")
    
    # Try to find train.csv in various locations
    possible_paths = [
        '/content/food_dataset/train.csv',
        './food_dataset/train.csv',
        '../food_dataset/train.csv',
        '/content/Food-RepViT/food_dataset/train.csv',
    ]
    
    for csv_path in possible_paths:
        if os.path.exists(csv_path):
            print(f"Found CSV: {csv_path}")
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                labels = sorted(df['Food_Label'].unique())
                
                if len(labels) != num_classes:
                    print(f"WARNING: CSV has {len(labels)} labels but model expects {num_classes}")
                    print("Using only the first labels or padding with generic names")
                    
                    if len(labels) > num_classes:
                        labels = labels[:num_classes]
                    else:
                        # Pad with generic labels
                        for i in range(len(labels), num_classes):
                            labels.append(f"Class_{i}")
                
                idx_to_label = {idx: label for idx, label in enumerate(labels)}
                print(f"✓ Loaded {len(labels)} food categories:")
                for idx, label in list(idx_to_label.items())[:5]:
                    print(f"  {idx}: {label}")
                if len(idx_to_label) > 5:
                    print(f"  ... and {len(idx_to_label) - 5} more")
                
                return idx_to_label
                
            except Exception as e:
                print(f"Error reading CSV: {e}")
    
    # Fallback: create generic labels
    print("WARNING: Could not load label names from CSV")
    print("Using generic labels: Class_0, Class_1, ...")
    idx_to_label = {i: f"Class_{i}" for i in range(num_classes)}
    return idx_to_label


def predict_image(model, image_path, idx_to_label, device, top_k=5):
    """
    Predict food category for an image
    
    Args:
        model: Loaded model
        image_path: Path to image
        idx_to_label: Label mapping dictionary
        device: torch device
        top_k: Number of top predictions
        
    Returns:
        List of (label, confidence) tuples
    """
    # Setup transform
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    
    # Load and preprocess image
    try:
        img = Image.open(image_path).convert('RGB')
        print(f"✓ Image loaded: {img.size}")
    except Exception as e:
        print(f"ERROR loading image: {e}")
        return None
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_probs, top_indices = torch.topk(probabilities, min(top_k, len(idx_to_label)))
    
    # Format results
    results = []
    for prob, idx in zip(top_probs, top_indices):
        label = idx_to_label[idx.item()]
        confidence = prob.item() * 100
        results.append((label, confidence))
    
    return results


def main():
    parser = argparse.ArgumentParser('Food Classification Inference')
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--image', required=True, type=str,
                        help='Path to input image')
    parser.add_argument('--model', default='repvit_m0_9', type=str,
                        help='Model architecture name')
    parser.add_argument('--top-k', default=5, type=int,
                        help='Number of top predictions to show')
    
    args = parser.parse_args()
    
    # Verify files exist
    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    if not Path(args.image).exists():
        print(f"ERROR: Image not found: {args.image}")
        sys.exit(1)
    
    print("=" * 70)
    print("FOOD CLASSIFICATION - SINGLE IMAGE INFERENCE")
    print("=" * 70)
    
    # Load model and labels
    model, idx_to_label, num_classes, device = load_model_and_labels(
        args.checkpoint, args.model
    )
    
    # Make prediction
    print("\n" + "=" * 70)
    print(f"Analyzing image: {args.image}")
    print("=" * 70)
    
    results = predict_image(model, args.image, idx_to_label, device, args.top_k)
    
    if results is None:
        print("Failed to process image")
        sys.exit(1)
    
    # Display results
    print(f"\nTop-{args.top_k} Predictions:")
    print("-" * 70)
    for rank, (label, confidence) in enumerate(results, 1):
        bar_length = int(confidence / 2)  # Scale to 50 chars max
        bar = "█" * bar_length
        print(f"{rank}. {label:30s} {confidence:6.2f}%  {bar}")
    
    print("=" * 70)
    
    # Return top prediction
    return results[0]


if __name__ == '__main__':
    main()