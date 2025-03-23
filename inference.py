import argparse
import torch
from PIL import Image
from torchvision import transforms
from model import ViT
import matplotlib.pyplot as plt

def load_model(model_path, num_classes=4):
    """Load a pre-trained ViT model"""
    model = ViT(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image_path):
    """Load and preprocess an image for inference"""
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    # Apply transformations
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor, img

def predict(model, img_tensor, device='cpu'):
    """Run inference on an image"""
    model.to(device)
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence, probabilities.cpu().numpy()[0]

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ViT Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='vit_cifar10.pth', help='Path to model checkpoint')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
    print(f"Using device: {device}")
    
    # Class labels
    class_names = ['airplane', 'automobile', 'bird', 'cat']
    
    # Load model
    print("Loading model...")
    model = load_model(args.model)
    
    # Preprocess image
    print(f"Processing image: {args.image}")
    img_tensor, original_img = preprocess_image(args.image)
    
    # Run inference
    predicted_class, confidence, all_probs = predict(model, img_tensor, device)
    
    # Display results
    print(f"Predicted class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence*100:.2f}%")
    
    # Display all class probabilities
    print("\nClass probabilities:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {all_probs[i]*100:.2f}%")
    
    # Display the image with prediction
    plt.figure(figsize=(8, 6))
    plt.imshow(original_img)
    plt.title(f"Prediction: {class_names[predicted_class]} ({confidence*100:.2f}%)")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()