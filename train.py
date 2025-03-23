import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from model import ViT
from custom_data import train_loader, test_loader
from engine import train
import argparse


def save_plots(results, save_dir='.'):
    """Save loss and accuracy plots"""
    # Create plots directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(results['train_loss'], label='Train Loss')
    plt.plot(results['test_loss'], label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(results['train_acc'], label='Train Accuracy')
    plt.plot(results['test_acc'], label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    print(f"Plots saved to {os.path.join(save_dir, 'training_metrics.png')}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Vision Transformer')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--model_path', type=str, default='vit_cifar10.pth', help='Path to save model')
    parser.add_argument('--plots_dir', type=str, default='.', help='Directory to save plots')
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(132)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(132)
    
    # Initialize ViT model
    print("Initializing ViT model...")
    vit = ViT(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=4,  # For the 4 classes in our subset
        embedding_dim=768,
        num_heads=12,
        num_transformer_layer=12
    )
    
    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=vit.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=0.5
    )
    
    # Train the model
    print(f"Starting training for {args.epochs} epochs...")
    results = train(
        model=vit,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=args.epochs,
        device=device
    )
    
    # Save the trained model
    torch.save(vit.state_dict(), args.model_path)
    print(f"Model saved as {args.model_path}")
    
    # Plot and save training metrics
    save_plots(results, args.plots_dir)

if __name__ == "__main__":
    main()