import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2
from covid_xray_classifier import SimpleCNN, BASE_DIR

# Directory settings
MODEL_PATH = os.path.join(BASE_DIR, 'covid_custom_cnn_model.pth')
DATA_DIR = os.path.join(BASE_DIR, 'etiquetadas')
RESULTS_DIR = os.path.join(BASE_DIR, 'gradcam_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Device configuration
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Image preprocessing
IMG_SIZE = 224
CROP_SIZE = int(IMG_SIZE * 0.7)  # Same crop size as in training

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.CenterCrop(CROP_SIZE),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# GradCAM implementation
class GradCAM:
    """
    Implements Gradient-weighted Class Activation Mapping (Grad-CAM) for visualizing
    which parts of an image are important for classification.
    """
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to capture gradients and activations
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_heatmap(self, input_image, target_class=None):
        """
        Generate a GradCAM heatmap for the input image
        
        Args:
            input_image: Input tensor (preprocessed image)
            target_class: Target class index for the heatmap (None = use predicted class)
            
        Returns:
            heatmap: Numpy array with the heatmap
            class_idx: The class index used for the heatmap
        """
        # Forward pass to get the prediction
        output = self.model(input_image)
        
        # If no target class is provided, use the predicted class
        if target_class is None:
            target_class = torch.argmax(output).item()
            
        # Clear existing gradients
        self.model.zero_grad()
        
        # One-hot encode the target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        
        # Backward pass to get gradients
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get weights by global average pooling the gradients
        weights = F.adaptive_avg_pool2d(self.gradients, 1)
        
        # Generate the class activation map (weighted sum of feature maps)
        cam = torch.zeros_like(self.activations[0, 0]).to(DEVICE)
        for i, w in enumerate(weights[0]):
            cam += w * self.activations[0, i]
        
        # Apply ReLU to the CAM (positive contributions only)
        cam = F.relu(cam)
        
        # Normalize the CAM
        cam = cam - torch.min(cam)
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
            
        # Convert to numpy and resize to original image dimensions
        heatmap = cam.cpu().numpy()
        
        return heatmap, target_class
    
def load_model():
    """Load the trained model from checkpoint"""
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Get the number of classes from the checkpoint
    class_to_idx = checkpoint.get('class_to_idx', None)
    num_classes = len(class_to_idx) if class_to_idx else 2
    
    # Create and load model
    model = SimpleCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    print(f"Loaded model with {num_classes} classes.")
    print(f"Class mapping: {class_to_idx}")
    
    return model, class_to_idx

def preprocess_image(image_path):
    """Load and preprocess an image for model input"""
    image = Image.open(image_path).convert('L')  # Load as grayscale
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    return image, input_tensor

def visualize_gradcam(image_path, model, class_to_idx, target_class=None, save=True):
    """
    Generate and visualize GradCAM for an image
    
    Args:
        image_path: Path to the image file
        model: The trained model
        class_to_idx: Dictionary mapping class names to indices
        target_class: Target class index (None = use predicted class)
        save: Whether to save the visualization
    """
    # Load and preprocess image
    original_image, input_tensor = preprocess_image(image_path)
    
    # Create GradCAM instance targeting the last convolutional layer
    gradcam = GradCAM(model, target_layer=model.conv5[-2])  # Target ReLU in last conv block
    
    # Generate heatmap
    heatmap, predicted_idx = gradcam.generate_heatmap(input_tensor, target_class)
    
    # Resize heatmap to match original image size
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    
    # Convert heatmap to RGB colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
    
    # Convert original image to numpy array
    orig_np = np.array(original_image.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    
    # Create 3-channel grayscale image
    if len(orig_np.shape) == 2:  # If single channel
        orig_np = np.stack([orig_np] * 3, axis=2)
    
    # Create the superimposed image (heatmap overlay)
    superimposed = 0.6 * orig_np + 0.4 * heatmap_colored
    
    # Get the class name
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_name = idx_to_class.get(predicted_idx, f"Unknown (idx: {predicted_idx})")
    
    # Create the figure for visualization
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    axs[0].imshow(orig_np, cmap='bone')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    # Plot heatmap
    axs[1].imshow(heatmap_colored)
    axs[1].set_title('GradCAM Heatmap')
    axs[1].axis('off')
    
    # Plot superimposed image
    axs[2].imshow(superimposed)
    axs[2].set_title('GradCAM Overlay')
    axs[2].axis('off')
    
    # Add suptitle with class information
    plt.suptitle(f"Class: {class_name}", fontsize=14)
    
    plt.tight_layout()
    
    # Save figure if requested
    if save:
        filename = os.path.basename(image_path)
        base_name = os.path.splitext(filename)[0]
        save_path = os.path.join(RESULTS_DIR, f"{base_name}_gradcam.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"GradCAM visualization saved to {save_path}")
    
    return fig

def process_sample_images(model, class_to_idx, num_samples=3):
    """Process sample images from each class"""
    # Get class directories
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    for idx, class_name in idx_to_class.items():
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"Directory not found: {class_dir}")
            continue
        
        # Get image files (exclude hidden files)
        image_files = [f for f in os.listdir(class_dir) 
                      if not f.startswith('.') and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"No images found in {class_dir}")
            continue
            
        # Select sample images
        samples = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
        
        print(f"\nProcessing {len(samples)} sample images for class '{class_name}':")
        for sample in samples:
            image_path = os.path.join(class_dir, sample)
            print(f"  - {os.path.basename(image_path)}")
            
            # Generate GradCAM visualization
            visualize_gradcam(image_path, model, class_to_idx, target_class=idx)
            plt.close()  # Close figure to prevent display in notebooks/scripts

def main():
    """Main function to run GradCAM visualization"""
    print("Loading model...")
    model, class_to_idx = load_model()
    
    print("\nGenerating GradCAM visualizations for sample images...")
    process_sample_images(model, class_to_idx, num_samples=6)
    
    print(f"\nAll GradCAM visualizations have been saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
