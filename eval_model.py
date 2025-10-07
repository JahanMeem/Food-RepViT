import torch
import argparse
from pathlib import Path
import sys

# Fix for PyTorch 2.6+ checkpoint loading
torch.serialization.add_safe_globals([argparse.Namespace])

from timm.models import create_model
from custom_food_dataset import build_food_dataset
from engine import evaluate
import model
import utils


def get_args():
    parser = argparse.ArgumentParser('RepViT Model Evaluation')
    
    # Model settings
    parser.add_argument('--model', default='repvit_m0_9', type=str,
                        help='Model name')
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='Path to checkpoint file (.pth)')
    
    # Data settings
    parser.add_argument('--data-path', default='/content/food_dataset', type=str,
                        help='Path to dataset root directory')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', default=2, type=int,
                        help='Number of data loading workers')
    parser.add_argument('--input-size', default=224, type=int,
                        help='Input image size')
    
    # Transform settings (needed for dataset building)
    parser.add_argument('--color-jitter', type=float, default=0.4)
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1')
    parser.add_argument('--train-interpolation', type=str, default='bicubic')
    parser.add_argument('--reprob', type=float, default=0.25)
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)
    parser.add_argument('--finetune', default='', type=str)
    
    # Device
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device to use (cuda or cpu)')
    
    return parser.parse_args()


def main():
    args = get_args()
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    device = torch.device(args.device)
    
    print("=" * 70)
    print("REPVIT MODEL EVALUATION")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Data path: {args.data_path}")
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"\nERROR: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    # Build test dataset
    print("\n" + "-" * 70)
    print("Loading test dataset...")
    print("-" * 70)
    
    try:
        dataset_test, num_classes = build_food_dataset(is_train=False, args=args)
        print(f"✓ Test samples: {len(dataset_test)}")
        print(f"✓ Number of classes: {num_classes}")
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        sys.exit(1)
    
    # Create data loader
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # Create model
    print("\n" + "-" * 70)
    print("Creating and loading model...")
    print("-" * 70)
    
    try:
        model_net = create_model(
            args.model,
            num_classes=num_classes,
            pretrained=False,
        )
        print(f"✓ Model created: {args.model}")
        
        # Count parameters
        n_parameters = sum(p.numel() for p in model_net.parameters() if p.requires_grad)
        print(f"✓ Model parameters: {n_parameters / 1e6:.2f}M")
        
    except Exception as e:
        print(f"ERROR creating model: {e}")
        sys.exit(1)
    
    # Load checkpoint
    try:
        print(f"\nLoading checkpoint...")
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        
        # Check what's in the checkpoint
        checkpoint_keys = list(checkpoint.keys())
        print(f"✓ Checkpoint loaded")
        print(f"  Keys in checkpoint: {checkpoint_keys}")
        
        if 'epoch' in checkpoint:
            print(f"  Training epoch: {checkpoint['epoch']}")
        
        # Load model weights
        if 'model' in checkpoint:
            model_net.load_state_dict(checkpoint['model'])
            print(f"✓ Model weights loaded")
        else:
            print("ERROR: 'model' key not found in checkpoint")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if the checkpoint file is corrupted")
        print("2. Ensure the model architecture matches the checkpoint")
        print("3. Try loading with: torch.load(path, weights_only=False)")
        sys.exit(1)
    
    # Prepare model for inference
    print("\nPreparing model for inference...")
    utils.replace_batchnorm(model_net)  # Merge Conv-BN layers
    model_net.to(device)
    model_net.eval()
    print("✓ Model ready for inference")
    
    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATING MODEL ON TEST SET")
    print("=" * 70)
    
    try:
        test_stats = evaluate(data_loader_test, model_net, device)
        
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        print(f"Top-1 Accuracy:  {test_stats['acc1']:.2f}%")
        print(f"Top-5 Accuracy:  {test_stats['acc5']:.2f}%")
        print(f"Average Loss:    {test_stats['loss']:.4f}")
        print(f"Test Samples:    {len(dataset_test)}")
        print("=" * 70)
        
        # Save results to file
        results_file = Path(args.checkpoint).parent / 'evaluation_results.txt'
        with open(results_file, 'w') as f:
            f.write(f"Model: {args.model}\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Test samples: {len(dataset_test)}\n")
            f.write(f"Number of classes: {num_classes}\n")
            f.write(f"Top-1 Accuracy: {test_stats['acc1']:.2f}%\n")
            f.write(f"Top-5 Accuracy: {test_stats['acc5']:.2f}%\n")
            f.write(f"Average Loss: {test_stats['loss']:.4f}\n")
        
        print(f"\n✓ Results saved to: {results_file}")
        
        return test_stats
        
    except Exception as e:
        print(f"ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()