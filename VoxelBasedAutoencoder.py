import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ------------------------------------------------
#               Dataset
# ------------------------------------------------
class Voxel128Dataset(Dataset):
    def __init__(self, root_dir, max_files=None):
        super().__init__()
        self.root_dir = root_dir
        self.files = []
        
        all_files = [f for f in os.listdir(root_dir) if f.lower().endswith(".npy")]
        
        if max_files is not None:
            all_files = all_files[:max_files]
            
        for f in all_files:
            try:
                # Don't load the full array here, just check if it exists
                file_path = os.path.join(root_dir, f)
                if os.path.exists(file_path):
                    self.files.append(f)
            except Exception as e:
                print(f"Error checking {f}: {e}")

        print(f"Dataset initialized with {len(self.files)} files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.files[idx])
        arr = np.load(path).astype(np.float32)
        # shape => (128,128,128)
        # add channel dim => (1,128,128,128)
        return torch.from_numpy(arr).unsqueeze(0)


# ------------------------------------------------
#   3D Autoencoder with fixed kernel=3, pad=1
# ------------------------------------------------
class FinalAutoencoder3D(nn.Module):
    """
    Autoencoder for 128^3 volumes.
    - kernel_size=3, padding=1, stride=2 in each downsample block => 128->64->32->16->8->4
    - device attribute is stored automatically
    """
    def __init__(self, latent_dim=256):  # Set to specified latent_dim
        super().__init__()
        self._device = torch.device("cpu")

        # Hardcoded kernel=3, pad=1, stride=2
        kernel_size = 3
        padding = 1

        # -- Encoder (4 downsampling blocks) --
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=kernel_size, stride=2, padding=padding), #stride 2 zorgt voor halvering in dim, 16 feature channels betekent 16 3D features, hier van grootte 64x64x64
            nn.PReLU(), # elke filter leert verschillende patronen van de input
            nn.Conv3d(16, 16, kernel_size=kernel_size, stride=1, padding=padding), 
            nn.PReLU(),

            nn.Conv3d(16, 32, kernel_size=kernel_size, stride=2, padding=padding), #Input: 16×64×64×64, Output: 32×32×32×32, gehalveerd want stride=2
            nn.PReLU(),
            nn.Conv3d(32, 32, kernel_size=kernel_size, stride=1, padding=padding),
            nn.PReLU(),

            nn.Conv3d(32, 64, kernel_size=kernel_size, stride=2, padding=padding), #32×32×32×32, Output: 64×16×16×16, gehalveerd want stride=2
            nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=kernel_size, stride=1, padding=padding),
            nn.PReLU(),

            nn.Conv3d(64, 128, kernel_size=kernel_size, stride=2, padding=padding), #Input: 64×16×16×16, Output: 128×8×8×8, gehalveerd want stride=2
            nn.PReLU(), # betekent 128 feature channels, dus 128 3D features, hier van grootte 8x8x8
            nn.Conv3d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.PReLU(),
        )
        # het feit dat we op het einde van de encoder 128 feature channels hebben en de input dimensie ook 128 is, is een soort van toeval
        # het is normaal dat je begint met het gaan van 1 feature channel naar 16, 32, 64, 128, en dat je dan elke keer de feature channels verdubbelt
        # de input dimensie halveert elke block, en dat is ook wat we willen, dus dat is een goede keuze
        # we hadden ook met 8 input channels kunnen beginnen en dan hadden we niet toevallig evenveel feature channels als input dimensie
        # Flatten => fc => latent
        self.fc_mu = nn.Linear(128 * 8 * 8 * 8, latent_dim) #Input: 128×8×8×8 → flattened to 65,536, Output: latent_dim (768)
        self.fc_dec = nn.Linear(latent_dim, 128 * 8 * 8 * 8)

        # -- Decoder (5 upsampling blocks) --
        self.decoder = nn.Sequential(
            # Block 1: 8³ -> 16³
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=kernel_size, padding=padding),
            nn.PReLU(),
            
            # Block 2: 16³ -> 32³
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
            nn.PReLU(),
            nn.Conv3d(32, 32, kernel_size=kernel_size, padding=padding),
            nn.PReLU(),
            
            # Block 3: 32³ -> 64³
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),
            nn.PReLU(),
            nn.Conv3d(16, 16, kernel_size=kernel_size, padding=padding),
            nn.PReLU(),
            
            # Block 4: 64³ -> 128³
            nn.ConvTranspose3d(16, 8, kernel_size=2, stride=2),
            nn.PReLU(),
            nn.Conv3d(8, 8, kernel_size=kernel_size, padding=padding),
            nn.PReLU(),
            
            # Block 5: Additional refinement at 128³ resolution
            nn.Conv3d(8, 4, kernel_size=kernel_size, padding=padding),
            nn.PReLU(),
            nn.Conv3d(4, 1, kernel_size=1),  # Final 1×1 convolution to get single channel output
            # No activation for BCEWithLogitsLoss
        )

    @property
    def device(self):
        return self._device

    def to(self, device):
        super().to(device)
        self._device = device
        return self

    def encode(self, x):
        x_enc = self.encoder(x)
        z = self.fc_mu(x_enc.flatten(1))
        return z

    def decode(self, z):
        x = self.fc_dec(z)
        x = x.view(-1, 128, 8, 8, 8)
        x = self.decoder(x)
        return x

    def forward(self, x):
        return self.decode(self.encode(x))


# ---------------------------
#        Train Helpers
# ---------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    # Activates training-specific behaviors like dropout and batch normalization
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for vox in loader:
        # Move data to device and ensure it's the right type
        vox = vox.to(device, non_blocking=True)
        batch_size = vox.size(0)
        
        # Clear gradients
        # Using set_to_none=True deallocates memory rather than just zeroing gradients
        optimizer.zero_grad(set_to_none=True)  # More memory efficient
        
        # Forward pass
        # Passes input through the autoencoder to get reconstructed voxels
        recon = model(vox) #uses an existing instance of FinalAutoencoder3D and This invokes a special Python method called __call__, which triggers the forward method of the model
        # Computes reconstruction error between original and reconstructed voxels
        loss = criterion(recon, vox)
        
        # Backward pass and optimize
        #  Calculates gradients of the loss with respect to model parameters
        loss.backward()
        # Adjusts model weights based on gradients and optimization algorithm
        optimizer.step()
        
        # Update statistics
        # Adds weighted batch loss to running total
        total_loss += loss.item() * batch_size
        # Updates total processed sample count
        total_samples += batch_size
        
        # Clean up to free memory, Explicitly removes tensors to free memory
        del vox, recon, loss
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    # Calculates and returns mean loss per sample for this epoch
    return total_loss / total_samples

def eval_epoch(model, loader, criterion, device):
    # Set model to evaluation mode - disables dropout and uses running statistics for batch norm
    model.eval()
    
    # Initialize tracking variables
    total_loss = 0.0
    total_samples = 0
    
    # Disable gradient computation to save memory during evaluation
    with torch.no_grad():
        for vox in loader:
            # Transfer batch data to appropriate device (GPU/CPU)
            vox = vox.to(device, non_blocking=True)
            batch_size = vox.size(0)
            
            # Forward pass only (no parameter updates during evaluation)
            recon = model(vox)
            # Calculate reconstruction error between original and predicted voxels
            loss = criterion(recon, vox)
            
            # Accumulate weighted loss statistics
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Explicitly free memory to avoid GPU memory fragmentation
            del vox, recon, loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    # Return average loss per sample across the entire validation set
    return total_loss / total_samples


# ---------------------------
#        Main Training
# ---------------------------
def train_autoencoder(data_dir, output_dir, epochs=500, max_files=None, patience=5):
    """
    Train the autoencoder with specified hyperparameters.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Fixed hyperparameters from optimization with optuna
    batch_size = 8
    lr = 0.00023599965647181094  
    latent_dim = 256  
    
    # Set up model, loss function, and optimizer
    model = FinalAutoencoder3D(latent_dim=latent_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Load dataset (max_files=None will load all files)
    full_dataset = Voxel128Dataset(data_dir, max_files=max_files)
    
    # Split dataset
    ds_size = len(full_dataset)
    train_size = int(0.9 * ds_size) ## 90% for training because the dataset is large and we want to train on as much data as possible
    val_size = ds_size - train_size
    
    # Create training and validation datasets
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=True)
    
    print(f"Training with {train_size} samples, validating with {val_size} samples")
    print(f"Using hyperparameters: batch_size={batch_size}, lr={lr}, latent_dim={latent_dim}")
    
    # Training loop
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    training_history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'time': [],
        'early_stopped': False
    }
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train and evaluate
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device) #updates model weights based on training data
        val_loss = eval_epoch(model, val_loader, criterion, device)
        
        # Record time
        epoch_time = time.time() - start_time
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Time: {epoch_time:.2f}s")
        
        # Save history
        training_history['epoch'].append(epoch + 1)
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['time'].append(epoch_time)
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Create model state dictionary
            model_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'hyperparams': {
                    'batch_size': batch_size,
                    'lr': lr,
                    'latent_dim': latent_dim
                }
            }
            
            # Save only when there's improvement, overwriting the previous best model
            best_model_path = os.path.join(output_dir, "autoencoder_best.pt")
            torch.save(model_state, best_model_path)
            
            print(f"New best model saved with val_loss: {best_val_loss:.6f}, at eth epoch {epoch+1}")
            epochs_without_improvement = 0  # Reset counter
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs")
            
            # Early stopping check
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                training_history['early_stopped'] = True
                break
        
        # Save training history
        np.save(os.path.join(output_dir, "training_history.npy"), training_history)
    
    # Final message
    if training_history['early_stopped']:
        print(f"Training stopped early due to no improvement for {patience} epochs")
    else:
        print(f"Training completed after {epochs} epochs")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    return model


# ---------------------------
#        Evaluation Functions
# ---------------------------
def visualize_reconstruction(model, voxel_path, output_dir=None, threshold=0.5):
    """
    Visualize the original voxel and its reconstruction.
    
    Args:
        model: Trained autoencoder model
        voxel_path: Path to .npy file containing voxel data
        output_dir: Directory to save visualizations
        threshold: Threshold for binary voxel reconstruction
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Load voxel data
    voxel_data = np.load(voxel_path).astype(np.float32)
    voxel_tensor = torch.from_numpy(voxel_data).unsqueeze(0).unsqueeze(0).to(device)
    
    # Get reconstruction
    with torch.no_grad():
        recon_tensor = model(voxel_tensor)
        recon_data = torch.sigmoid(recon_tensor).squeeze().cpu().numpy()
    
    # Apply threshold for binary reconstruction
    binary_recon = (recon_data > threshold).astype(np.float32)
    
    # Create directory for output if it doesn't exist
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Plot 2D slices from middle of each axis
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original voxel
    axs[0, 0].imshow(voxel_data[voxel_data.shape[0]//2, :, :], cmap='gray')
    axs[0, 0].set_title('Original - XY Plane (Middle Slice)')
    axs[0, 1].imshow(voxel_data[:, voxel_data.shape[1]//2, :], cmap='gray')
    axs[0, 1].set_title('Original - XZ Plane (Middle Slice)')
    axs[0, 2].imshow(voxel_data[:, :, voxel_data.shape[2]//2], cmap='gray')
    axs[0, 2].set_title('Original - YZ Plane (Middle Slice)')
    
    # Reconstructed voxel
    axs[1, 0].imshow(binary_recon[binary_recon.shape[0]//2, :, :], cmap='gray')
    axs[1, 0].set_title('Reconstructed - XY Plane (Middle Slice)')
    axs[1, 1].imshow(binary_recon[:, binary_recon.shape[1]//2, :], cmap='gray')
    axs[1, 1].set_title('Reconstructed - XZ Plane (Middle Slice)')
    axs[1, 2].imshow(binary_recon[:, :, binary_recon.shape[2]//2], cmap='gray')
    axs[1, 2].set_title('Reconstructed - YZ Plane (Middle Slice)')
    
    plt.tight_layout()
    
    # Save or show plot
    if output_dir is not None:
        filename = os.path.basename(voxel_path).split('.')[0]
        plt.savefig(os.path.join(output_dir, f"{filename}_reconstruction.png"))
        plt.close()
    else:
        plt.show()
    
    # Calculate reconstruction metrics
    iou = calculate_iou(voxel_data, binary_recon)
    
    # Create 3D visualization if needed
    # visualize_3d(voxel_data, binary_recon, output_dir, filename)
    
    return iou

def calculate_iou(original, reconstructed):
    """
    Calculate Intersection over Union (IoU) between original and reconstructed voxels.
    """
    intersection = np.logical_and(original > 0, reconstructed > 0).sum()
    union = np.logical_or(original > 0, reconstructed > 0).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union

def evaluate_model(model, data_dir, output_dir=None, num_samples=10, threshold=0.5):
    """
    Evaluate the autoencoder model on a set of samples.
    """
    # Get list of files
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(".npy")]
    
    # Select random samples
    if len(all_files) > num_samples:
        sample_files = np.random.choice(all_files, num_samples, replace=False)
    else:
        sample_files = all_files
    
    # Evaluate each sample
    iou_scores = []
    
    for voxel_path in sample_files:
        iou = visualize_reconstruction(model, voxel_path, output_dir, threshold)
        iou_scores.append(iou)
        print(f"File: {os.path.basename(voxel_path)}, IoU: {iou:.4f}")
    
    # Calculate average metrics
    avg_iou = np.mean(iou_scores)
    
    print(f"\nEvaluation Results:")
    print(f"Average IoU: {avg_iou:.4f}")
    
    results = {
        'iou_scores': iou_scores,
        'avg_iou': avg_iou,
        'sample_files': [os.path.basename(f) for f in sample_files]
    }
    
    # Save results
    if output_dir is not None:
        np.save(os.path.join(output_dir, "evaluation_results.npy"), results)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a 3D autoencoder for voxel data")
    parser.add_argument("--data_dir", type=str, 
                        default="C:\\Users\\maart\\Documents\\programmeren\\Python\\AutoEncoders\\NumpyVoxels",
                        help="Directory containing .npy voxel files")
    parser.add_argument("--output_dir", type=str, default="./model_checkpoints", 
                        help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--max_files", type=int, default=None, 
                        help="Maximum number of files to use for training (None to use all)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument("--evaluate", action="store_true", 
                        help="Evaluate model reconstruction after training")
    parser.add_argument("--eval_samples", type=int, default=10, 
                        help="Number of samples to use for evaluation")
    parser.add_argument("--mode", choices=["train", "eval"], default="train", 
                        help="Mode: train or evaluate")
    parser.add_argument("--model_path", type=str, help="Path to model for evaluation mode")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        # Train the autoencoder with early stopping
        model = train_autoencoder(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            max_files=args.max_files,
            patience=args.patience
        )
        
        # u can evaluate the model if u want
        if args.evaluate:
            best_model_path = os.path.join(args.output_dir, "autoencoder_best.pt")
            eval_dir = os.path.join(args.output_dir, "evaluation")
            evaluate_model(model, args.data_dir, eval_dir, args.eval_samples)
    
    elif args.mode == "eval":
        if args.model_path is None:
            print("Error: --model_path is required for evaluation mode")
            exit(1)
        
        # Load model
        checkpoint = torch.load(args.model_path, map_location='cpu')
        latent_dim = checkpoint['hyperparams']['latent_dim']
        model = FinalAutoencoder3D(latent_dim=latent_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate model
        eval_dir = os.path.join(os.path.dirname(args.model_path), "evaluation")
        evaluate_model(model, args.data_dir, eval_dir, args.eval_samples)