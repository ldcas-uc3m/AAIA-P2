import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score
import torch.multiprocessing
import time  # Import time module for measuring execution time

# Fix multiprocessing issues on macOS - prevents "too many open files" errors
torch.multiprocessing.set_sharing_strategy('file_system')

SEED = 12345
# Set random seeds for reproducibility - ensures same results on different runs
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# Define directory paths for dataset and model storage
BASE_DIR = '/Users/jorge/clones/IA2/covid'
DATA_DIR = os.path.join(BASE_DIR, 'etiquetadas')
COVID_DIR = os.path.join(DATA_DIR, 'COVID-19')
NORMAL_DIR = os.path.join(DATA_DIR, 'Normal')
BACTERIAL_DIR = os.path.join(DATA_DIR, 'Bacterial')
MODEL_PATH = os.path.join(BASE_DIR, 'covid_custom_cnn_model.pth')  # Updated model path

# Check if dataset exists to provide a helpful error message if missing
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

# Print dataset statistics to give feedback on data availability
if os.path.exists(COVID_DIR):
    covid_images = [f for f in os.listdir(COVID_DIR) if not f.startswith('.')]  # Exclude hidden files
    print(f"COVID images: {len(covid_images)}")
if os.path.exists(NORMAL_DIR):
    normal_images = [f for f in os.listdir(NORMAL_DIR) if not f.startswith('.')]
    print(f"Normal images: {len(normal_images)}")
if os.path.exists(BACTERIAL_DIR):
    bacterial_images = [f for f in os.listdir(BACTERIAL_DIR) if not f.startswith('.')]
    print(f"Bacterial images: {len(bacterial_images)}")

IMG_SIZE = 224  # Input image size
BATCH_SIZE = 32  # Number of images processed in each batch
EPOCHS = 30 # Number of training epochs (will stop early if no improvement)
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
NUM_WORKERS = 0

# Use Apple Metal Performance Shaders if available, otherwise fall back to CPU
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Data transformations for training with augmentation to improve model generalization
train_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),  # Resize the smaller side to IMG_SIZE while preserving aspect ratio
    transforms.CenterCrop((IMG_SIZE, IMG_SIZE)),  # Then crop to get a square image
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.RandomAffine(
        degrees=15,  # Rotation range: -15 to +15 degrees
        scale=(1.05, 1.15)  # Scale range: 95% to 105%
    ),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize for grayscale images
])

# Data transformations for validation (no augmentation to get accurate evaluation)
val_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),  # Resize the smaller side to IMG_SIZE while preserving aspect ratio
    transforms.CenterCrop((IMG_SIZE, IMG_SIZE)),  # Then crop to get a square image
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize([0.5], [0.5])  # Normalize for grayscale images
])

# Load the full dataset for training and validation
full_train_dataset = ImageFolder(DATA_DIR, transform=train_transforms)
full_val_dataset = ImageFolder(DATA_DIR, transform=val_transforms)

# Print class information
class_to_idx = full_train_dataset.class_to_idx
print("Class mapping:", class_to_idx)

# Get class names for later reference
idx_to_class = {v: k for k, v in class_to_idx.items()}
class_names = {idx: name for idx, name in idx_to_class.items()}
print("Classes:", class_names)

# Calculate split sizes
train_size = int((1 - VALIDATION_SPLIT) * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

# Split the datasets
train_dataset, _ = random_split(
    full_train_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

_, val_dataset = random_split(
    full_val_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Create dataloaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Define a simple CNN architecture from scratch
class SimpleCNN(nn.Module):
    """
    A custom CNN architecture built from scratch for X-ray image classification
    """
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        # First convolutional block - changed input channels from 3 to 1 for grayscale
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fourth convolutional block
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fifth convolutional block
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate the flattened feature size
        self.feature_size = 256 * 7 * 7
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Add dropout for regularization
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Apply convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # Apply classifier
        x = self.classifier(x)
        return x

# Build the model using our custom CNN
def build_model(num_classes=2):
    """
    Create a simple CNN model from scratch
    
    Args:
        num_classes: Number of output classes (2 for our binary classification)
        
    Returns:
        A PyTorch model with custom CNN architecture
    """
    model = SimpleCNN(num_classes=num_classes)
    return model

# New function: Handles training for a single epoch
def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch
    
    Args:
        model: The neural network model
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimization algorithm
        device: Device to run computations on (CPU, CUDA, or MPS)
        
    Returns:
        epoch_loss: Average loss for this epoch
        epoch_f1: F1 score for this epoch
    """
    # Set model to training mode (enables dropout, batch norm updates)
    model.train()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Iterate over data batches
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Zero the parameter gradients to prevent accumulation
        optimizer.zero_grad()
        
        # Forward pass to get outputs and calculate loss
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
    
        # Track batch statistics
        running_loss += loss.item() * inputs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate epoch statistics
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, epoch_f1

def validate_epoch(model, dataloader, criterion, device):
    """
    Validate the model for one epoch
    
    Args:
        model: The neural network model
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to run computations on (CPU, CUDA, or MPS)
        
    Returns:
        epoch_loss: Average loss for this epoch
        epoch_f1: F1 score for this epoch
    """
    model.eval()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Iterate over data batches without computing gradients
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass only
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            # Track batch statistics
            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate epoch statistics
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, epoch_f1

def train_model(model, dataloaders, criterion, optimizer, num_epochs, patience=5, min_delta=0.001):
    """
    Train and validate the model across multiple epochs with early stopping
    
    Args:
        model: The neural network model to train
        dataloaders: Dictionary containing training and validation DataLoaders
        criterion: Loss function
        optimizer: Optimization algorithm
        num_epochs: Maximum number of training epochs
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change in validation F1 to qualify as improvement
        
    Returns:
        best_model: The model with the best validation F1 score
        history: Dictionary containing training metrics for plotting
    """
    # History for plotting the training curves
    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': []
    }
    
    # Best model tracking variables
    best_model_wts = None
    best_f1 = 0.0
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        train_loss, train_f1 = train_epoch(
            model, dataloaders['train'], criterion, optimizer, DEVICE)
        print(f'Train Loss: {train_loss:.4f} F1: {train_f1:.4f}')
        
        # Validation phase
        val_loss, val_f1 = validate_epoch(
            model, dataloaders['val'], criterion, DEVICE)
        print(f'Validation Loss: {val_loss:.4f} F1: {val_f1:.4f}')
        
        # Save history for plotting
        history['train_loss'].append(train_loss)
        history['train_f1'].append(float(train_f1))
        history['val_loss'].append(val_loss)
        history['val_f1'].append(float(val_f1))
        
        # Check if performance improved
        if val_f1 > best_f1 + min_delta:
            print(f'Validation F1 improved from {best_f1:.4f} to {val_f1:.4f}')
            best_f1 = val_f1
            best_model_wts = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f'No improvement in validation F1 for {epochs_no_improve} epochs')
            
            # Check early stopping condition
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        print()  # Print empty line between epochs
    
    print(f'Best val F1: {best_f1:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

# Function to run the main training process
def run_training():
    """
    Main function that orchestrates the entire training process:
    1. Creates and initializes the model
    2. Performs training from scratch
    3. Evaluates the model and generates visualizations
    """
    # Record the start time
    start_time = time.time()
    
    # Create the model and move it to the compute device
    model = build_model()
    model = model.to(DEVICE)
    
    # Create loss function for classification and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training phase
    print("Training from scratch...")
    dataloaders = {'train': train_loader, 'val': val_loader}
    # Added patience=7 for early stopping
    model, history = train_model(model, dataloaders, criterion, optimizer, num_epochs=EPOCHS, patience=7)

    # Save the trained model to disk
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx  # Save class mapping for inference
    }, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Evaluate the model on validation set
    roc_auc = evaluate_model(model, val_loader)
    plot_training_history(history)
    
    # Calculate and print total training time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nModel evaluation complete. ROC-AUC: {roc_auc:.4f}")
    print(f"Total training time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d} (HH:MM:SS)")

# Evaluate the model and generate performance metrics
def evaluate_model(model, dataloader):
    """
    Evaluate the trained model and generate performance metrics and visualizations
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader for evaluation data
        
    Returns:
        roc_auc: Area under the ROC curve
    """
    model.eval()  # Set model to evaluation mode
    all_probs = []
    all_labels = []
    
    # Get the positive class index (COVID-19)
    covid_idx = class_to_idx.get('COVID-19', 0)  # Default to 0 if not found
    
    # Collect predictions without computing gradients
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            # Move to CPU before converting to numpy
            labels_cpu = labels.cpu().numpy()
            
            outputs = model(inputs)
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            
            # Extract probability for the positive class (COVID-19)
            covid_probs = probs[:, covid_idx]
            
            all_probs.extend(covid_probs)
            all_labels.extend((labels_cpu == covid_idx).astype(int))
    
    # Convert to numpy arrays for sklearn functions
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Generate binary predictions using 0.5 threshold
    binary_preds = (all_probs > 0.5).astype(int)
    
    # Calculate and print classification report (precision, recall, f1-score)
    print("\nClassification Report:")
    print(classification_report(all_labels, binary_preds))
    
    # Calculate and print confusion matrix
    cm = confusion_matrix(all_labels, binary_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot ROC curve to visualize model performance
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)  # False positive rate, true positive rate
    roc_auc = auc(fpr, tpr)  # Area under the curve
    
    # Calculate sensitivity at 85% specificity (which means fpr = 0.15)
    target_specificity = 0.85
    target_fpr = 1 - target_specificity  # False Positive Rate = 1 - Specificity
    
    # Check if we have perfect classification
    if roc_auc > 0.999:  # Using 0.999 instead of 1.0 to account for floating point imprecision
        print("\nPerfect classification detected.")
        print(f"At target specificity of {target_specificity:.2f}, sensitivity would be 1.00")
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        # Calculate true sensitivity from confusion matrix
        actual_sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        print(f"Actual sensitivity from confusion matrix: {actual_sensitivity:.4f}")
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # Diagonal line (random classifier)
        
        # For perfect classification, mark the corner point
        plt.plot(0, 1, 'ro', markersize=8, 
                label=f'Perfect classification (Sensitivity: 1.00, Specificity: 1.00)')
    else:
        # Find the threshold that gives closest to our target FPR
        diff = np.abs(fpr - target_fpr)
        idx = np.argmin(diff)
        
        # Get the sensitivity (TPR) at this point
        sensitivity_at_target = tpr[idx]
        actual_specificity = 1 - fpr[idx]
        
        print(f"\nAt {actual_specificity:.2f} specificity (target: {target_specificity:.2f}), sensitivity is: {sensitivity_at_target:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # Diagonal line (random classifier)
        
        # Mark the point for 85% specificity
        plt.plot(fpr[idx], tpr[idx], 'ro', markersize=8, 
                label=f'Sensitivity: {sensitivity_at_target:.2f} at {actual_specificity:.2f} specificity')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(BASE_DIR, 'roc_curve.png'))
    plt.close()
    
    return roc_auc

# Plot training history to visualize learning progress
def plot_training_history(history):
    """
    Generate plots of training and validation F1 score/loss over epochs
    
    Args:
        history: Dictionary containing training metrics
    """
    epochs_range = range(len(history['train_loss']))
    
    plt.figure(figsize=(12, 5))
    
    # Plot F1 score curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_f1'], label='Training F1 score')
    plt.plot(epochs_range, history['val_f1'], label='Validation F1 score')
    plt.legend(loc='lower right')
    plt.title('Training and Validation F1 Score')
    
    # Plot loss curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    # Save the figure to disk
    plt.savefig(os.path.join(BASE_DIR, 'training_history.png'))
    plt.close()

if __name__ == "__main__":
    # Display available classes in the dataset
    try:
        full_dataset = ImageFolder(DATA_DIR)
        print(f"Available classes: {full_dataset.classes}")
    except Exception as e:
        print(f"Error inspecting dataset: {e}")
    
    # Start the training process
    run_training()
