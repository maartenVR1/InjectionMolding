"""
MLP quality-prediction training pipeline
========================================
* scans   <MAIN_FOLDER>  where each product has its own sub-folder
* expects <product>/                               
      ├─ <product>.xlsx  (80 rows × 8 process cols + 3 target cols)
      └─ npy file  (voxel grid, 1 per product)

* learns an MLP that receives
      [ latent-vector (from pretrained 3-D AE)  ⨁  8 process parameters ]
  and predicts 3 quality metrics.

Splitting is **by product** (no leakage).

different trials use different loss functions but the validation and test metrics are always MSE to ensure
good comparison across trials.
"""

# ────────────────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────────────────
import os, math, random, logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import optuna
from optuna.trial import Trial
import json
import time

# ────────────────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# Device
# ────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# ────────────────────────────────────────────────────────────────────────────
# 3-D Auto-encoder  (unchanged)
# ────────────────────────────────────────────────────────────────────────────
class FinalAutoencoder3D(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self._device = torch.device("cpu")
        k, p = 3, 1
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, k, 2, p), nn.PReLU(),
            nn.Conv3d(16,16, k, 1, p), nn.PReLU(),
            nn.Conv3d(16,32, k, 2, p), nn.PReLU(),
            nn.Conv3d(32,32, k, 1, p), nn.PReLU(),
            nn.Conv3d(32,64, k, 2, p), nn.PReLU(),
            nn.Conv3d(64,64, k, 1, p), nn.PReLU(),
            nn.Conv3d(64,128,k, 2, p), nn.PReLU(),
            nn.Conv3d(128,128,k, 1, p), nn.PReLU(),
        )
        self.fc_mu  = nn.Linear(128*8*8*8, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 128*8*8*8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128,64, 2,2), nn.PReLU(),
            nn.Conv3d(64,64,k,1,p), nn.PReLU(),
            nn.ConvTranspose3d(64,32,2,2), nn.PReLU(),
            nn.Conv3d(32,32,k,1,p), nn.PReLU(),
            nn.ConvTranspose3d(32,16,2,2), nn.PReLU(),
            nn.Conv3d(16,16,k,1,p), nn.PReLU(),
            nn.ConvTranspose3d(16,8, 2,2), nn.PReLU(),
            nn.Conv3d(8,8,k,1,p),  nn.PReLU(),
            nn.Conv3d(8,4,k,1,p),  nn.PReLU(),
            nn.Conv3d(4,1,1),                     
        )

    @property
    def device(self): return self._device
    def to(self, device): super().to(device); self._device = device; return self
    def encode(self,x): return self.fc_mu(self.encoder(x).flatten(1)) # this is the latent space vector we want
    def decode(self,z):
        x = self.fc_dec(z).view(-1,128,8,8,8)
        return self.decoder(x)
    def forward(self,x): return self.decode(self.encode(x))

def load_autoencoder(path:str, device, latent_dim:int=256):
    chk = torch.load(path, map_location=device)
    sd  = chk["model_state_dict"] if "model_state_dict" in chk else chk
    model = FinalAutoencoder3D(latent_dim).to(device)
    model.load_state_dict(sd)
    model.eval()
    logger.info(f"Loaded AE from {path}")
    return model

# ────────────────────────────────────────────────────────────────────────────
# ##  DATASET  ##
# ────────────────────────────────────────────────────────────────────────────
class InjectionMoldingDataset(Dataset):
    """
    One row =  [ latent  ⨁  process params ]  →  targets
    So this is where the autoencoder is used to encode the 3D model, concatenates with the other parameters and then gets the targets.

    I tried to make sure it relatively easy to change this autoencoder out for another one, if you want to improve its performance or use a different architecture
    or type of autoencoder.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        voxel_root: Path,
        autoencoder: nn.Module,
        proc_cols, tgt_cols,
        scaler_proc: StandardScaler | None,
        scaler_tgt : StandardScaler | None,        
        device,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.proc_cols, self.tgt_cols = proc_cols, tgt_cols
        self.device, self.ae = device, autoencoder
        self.latent_dim = autoencoder.fc_mu.out_features

        # ---------- X : process
        if scaler_proc is not None:
            self.proc_arr = scaler_proc.transform(
                self.df[self.proc_cols].values.astype(float)
            )
        else:
            self.proc_arr = self.df[self.proc_cols].values.astype(float)

        # ---------- y : targets  (also scaled)             
        if scaler_tgt is not None:
            self.tgt_arr = scaler_tgt.transform(
                self.df[self.tgt_cols].values.astype(float)
            )
        else:
            self.tgt_arr = self.df[self.tgt_cols].values.astype(float)

        # cache latent vector per product
        self.latent_cache = {}  # Dictionary to store encoded 3D representations for each product
        with torch.no_grad():  # Disable gradient tracking for memory efficiency during encoding
            # Extract unique product-voxel file pairs to avoid redundant processing
            for _, row in self.df[["product_name", "cad_model_file"]].drop_duplicates().iterrows():
                product_folder = row["product_name"]   # The folder name for this product
                voxel_file = row["cad_model_file"]     # The .npy file containing 3D voxel data
                
                # Construct the full path to the voxel file using Path object's directory traversal
                # Format: /main_folder/product_folder/voxel_file.npy
                voxel_path = voxel_root / product_folder / voxel_file
                
                try:
                    # Load the 3D voxel grid from the .npy file
                    # - astype(np.float32): Convert to float32 for GPU compatibility
                    # - [None,None,...]: Add batch and channel dimensions, reshaping from (x,y,z) to (1,1,x,y,z)
                    #   required for 3D CNN input format
                    vox = np.load(voxel_path).astype(np.float32)[None,None,...]
                    
                    # Pass the voxel data through the encoder portion of the autoencoder:
                    # 1. Convert NumPy array to PyTorch tensor
                    # 2. Move tensor to the specified device (CPU/GPU)
                    # 3. Encode the 3D shape into a compact latent vector (e.g., 256 dimensions)
                    lat = self.ae.encode(torch.from_numpy(vox).to(device))
                    
                    # Store the encoded vector in the cache dictionary:
                    # - cpu(): Move back to CPU for storage efficiency
                    # - numpy(): Convert to NumPy for compatibility with future operations
                    # - squeeze(): Remove singleton dimensions, creating a 1D feature vector
                    self.latent_cache[voxel_file] = lat.cpu().numpy().squeeze()
                
                except FileNotFoundError:
                    # Detailed error logging if file is missing, with path information
                    logger.warning(f"File not found: {voxel_path}")
                    raise FileNotFoundError(f"Could not load voxel file {voxel_path}. Check directory structure.")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        lat  = self.latent_cache[row["cad_model_file"]]
        proc = self.proc_arr[idx]
        # Here the latent vector and process parameters are concatenated into a single input vector 
        x = np.concatenate([lat, proc])
        y = self.tgt_arr[idx]
        return {"input": torch.tensor(x, dtype=torch.float32),
                "target":torch.tensor(y, dtype=torch.float32)}

# ────────────────────────────────────────────────────────────────────────────
def make_activation(name:str):
    return {
        "relu":nn.ReLU(), "leaky_relu":nn.LeakyReLU(),
        "elu":nn.ELU(),   "tanh":nn.Tanh(), "selu":nn.SELU(),
    }[name]

def make_loss_function(name: str):
    """Create different loss functions based on name."""
    if name == "mse":
        return nn.MSELoss()
    elif name == "mae":
        return nn.L1Loss()
    elif name == "huber":
        return nn.HuberLoss(delta=1.0)
    elif name == "smooth_l1":
        return nn.SmoothL1Loss(beta=1.0)
    elif name == "log_cosh":
        # Custom implementation of Log-Cosh loss because it's not in PyTorch
        # Log-Cosh loss is defined as log(cosh(x)) = log((exp(x) + exp(-x)) / 2)
        # where x is the difference between predicted and true values
        def log_cosh_loss(y_pred, y_true):
            return torch.mean(torch.log(torch.cosh(y_pred - y_true)))
        return log_cosh_loss
    else:
        raise ValueError(f"Unknown loss function: {name}")

class MLPQualityPredictor(nn.Module):
    def __init__(self, input_dim, output_dim,
                 hidden_layers, neurons_per_layer,
                 activation:str, dropout:float):
        super().__init__()
        act = make_activation(activation)
        layers, dim_in = [], input_dim
        for i in range(hidden_layers):
            layers += [nn.Linear(dim_in, neurons_per_layer[i]), act]
            if dropout>0: layers.append(nn.Dropout(dropout))
            dim_in = neurons_per_layer[i]
        layers.append(nn.Linear(dim_in, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

# ────────────────────────────────────────────────────────────────────────────
# ##  TRAINER  ##
# ────────────────────────────────────────────────────────────────────────────
class MLPTrainer:
    def __init__(
        self,
        main_folder: str,
        autoencoder_path: str,
        latent_dim:int = 256,
        proc_cols = None,
        tgt_cols = None,
        batch_range = (8,64),
        random_state= 42,
        n_trials    = 200,
        study_name  = "MLP_quality_prediction",
        storage     = None,
        
        output_dir: str = "Thesis_System_Results",
    ):
        self.main_folder = Path(main_folder)
        self.auto_path   = autoencoder_path
        self.latent_dim  = latent_dim

        self.proc_cols = list(proc_cols)
        self.tgt_cols  = list(tgt_cols)

        self.batch_range  = batch_range
        self.random_state = random_state
        self.n_trials     = n_trials
        self.study_name   = study_name
        self.storage      = storage

        self.autoencoder = load_autoencoder(self.auto_path, DEVICE, latent_dim)
        self.full_df = self._load_all_products()
        self.train_df, self.val_df, self.test_df = self._split_by_product()

        # ---------- fit scalers                          
        self.scaler_proc = StandardScaler().fit(
            self.train_df[self.proc_cols].values.astype(float))
        self.scaler_tgt  = StandardScaler().fit(
            self.train_df[self.tgt_cols ].values.astype(float))

        # avoid div-by-zero for constant cols
        self.scaler_proc.scale_[self.scaler_proc.scale_==0] = 1.0
        self.scaler_tgt .scale_[self.scaler_tgt .scale_==0] = 1.0

        self.input_dim  = latent_dim + len(self.proc_cols)
        self.output_dim = len(self.tgt_cols)

        logger.info(f"Products  : {self.full_df['product_name'].nunique()}")
        logger.info(f"Train/Val/Test rows : "
                    f"{len(self.train_df)} / {len(self.val_df)} / {len(self.test_df)}")

        # Add output directory handling
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Export DataFrames to Excel for inspection
        self._export_dataframes_to_excel()

    # ───────────────────────────────── internal helpers ─────────────────────
    def _load_all_products(self) -> pd.DataFrame:
        """
        Load all product data from the main folder's subfolders.
        
        Expected structure:
        main_folder/
        ├── product1/
        │   ├── product1.xlsx  (contains process parameters and quality metrics)
        │   └── model.npy      (contains 3D voxel representation)
        ├── product2/
        │   ├── product2.xlsx
        │   └── model.npy
        └── ...
        
        Returns:
            pd.DataFrame: Combined dataframe with all products' data
        """
        rows = []  # List to store DataFrames for each product
        product = 1
        # Scan through each item in the main folder
        for prod_dir in self.main_folder.iterdir():
            # Skip if it's not a directory (e.g., if it's a file)
            if not prod_dir.is_dir(): continue
            
            # Find Excel files and NPY files in this product folder
            xl  = [f for f in prod_dir.iterdir() if f.suffix in (".xlsx", ".xls")]
            npy = [f for f in prod_dir.iterdir() if f.suffix == ".npy"]
            
            # Skip if either Excel or NPY file is missing
            if not xl or not npy:
                logger.warning(f"Skipping {prod_dir.name} (xlsx/npy missing)")
                continue
            
            '''
            # just checking if its loading all products
            print(f'loading product number {product}')  # Progress indicator
            product += 1
            '''

            # Load the Excel file into a DataFrame
            df = pd.read_excel(xl[0])  # Take the first Excel file if multiple exist
            
            # Add product identification columns
            df["product_name"]   = prod_dir.name       # Store folder name as product identifier
            # Only the numpy file name is stored here, InjectionMoldingDataset will load the actual file using this name
            df["cad_model_file"] = npy[0].name         # Store NPY filename for later 3D model loading
            
            # Add this product's data to our collection
            rows.append(df)
        
        # Make sure we found at least one valid product
        if not rows: raise RuntimeError("No valid products found")
        
        # Combine all product DataFrames into one large DataFrame
        return pd.concat(rows, ignore_index=True)

    def _split_by_product(self):
        """
        Split dataset by product name using a predefined canonical split from a JSON file.
        
        This approach ensures that:
        1. All data points from the same product are in the same split (train/val/test)
        2. The model is evaluated on completely unseen products during testing
        3. The split is consistent across different runs and experiments
        4. No data leakage occurs between splits
        
        Returns:
            tuple: (train_df, val_df, test_df) - Three pandas DataFrames with the split data
        """
        # Load the canonical product split configuration from external JSON file
        # This file contains predefined lists of product names for each split
        with open("canonical_product_split.json", "r") as f:
            product_split = json.load(f)
        
        # This helps document exactly which products were used for training/validation/testing
        logger.info("Using canonical product split:")
        logger.info(f"Train products: {len(product_split['train_products'])} - {product_split['train_products']}")
        logger.info(f"Val products: {len(product_split['val_products'])} - {product_split['val_products']}")
        logger.info(f"Test products: {len(product_split['test_products'])} - {product_split['test_products']}")
        
        # Filter the full dataset to create each split based on product names
        # Each resulting DataFrame contains ONLY rows where the product_name is in the corresponding list
        train_df = self.full_df[self.full_df["product_name"].isin(product_split["train_products"])].reset_index(drop=True)
        val_df = self.full_df[self.full_df["product_name"].isin(product_split["val_products"])].reset_index(drop=True)
        test_df = self.full_df[self.full_df["product_name"].isin(product_split["test_products"])].reset_index(drop=True)
        
        # Return the three split DataFrames
        return train_df, val_df, test_df

    def _make_dloaders(self, batch):
        kws=dict(num_workers=0,pin_memory=True)
        train_ds=InjectionMoldingDataset(
            self.train_df,self.main_folder,self.autoencoder,
            self.proc_cols,self.tgt_cols,
            self.scaler_proc,self.scaler_tgt,DEVICE)
    
        val_ds = InjectionMoldingDataset(
            self.val_df,self.main_folder,self.autoencoder,
            self.proc_cols,self.tgt_cols,
            self.scaler_proc,self.scaler_tgt,DEVICE)           
        
        test_ds = InjectionMoldingDataset(
            self.test_df,self.main_folder,self.autoencoder,
            self.proc_cols,self.tgt_cols,
            self.scaler_proc,self.scaler_tgt,DEVICE)

        return (
            DataLoader(train_ds,batch,shuffle=True,**kws),
            DataLoader(val_ds,batch,shuffle=False,**kws),
            DataLoader(test_ds,batch,shuffle=False,**kws),
        )

    def _export_dataframes_to_excel(self):
        """
        Export the train, validation and test DataFrames to Excel files for inspection.
        Files are saved in the output directory with descriptive names.
        """
        logger.info("Exporting DataFrames to Excel files for inspection...")
        
        # Save training data
        train_path = self.output_dir / f"{self.study_name}_train_data.xlsx"
        self.train_df.to_excel(train_path, index=False)
        logger.info(f"Training data ({len(self.train_df)} rows) saved to {train_path}")
        
        # Save validation data if it exists
        if not self.val_df.empty:
            val_path = self.output_dir / f"{self.study_name}_val_data.xlsx"
            self.val_df.to_excel(val_path, index=False)
            logger.info(f"Validation data ({len(self.val_df)} rows) saved to {val_path}")
        
        # Save test data
        test_path = self.output_dir / f"{self.study_name}_test_data.xlsx"
        self.test_df.to_excel(test_path, index=False)
        logger.info(f"Test data ({len(self.test_df)} rows) saved to {test_path}")

    # ─────────────────────────────────  Optuna objective  ───────────────────
    def _objective(self, trial:Trial):
        
        hl  = trial.suggest_int("hidden_layers",1,10)
        neu = [trial.suggest_int(f"n{i}",16,512,log=True) for i in range(hl)]
        act = trial.suggest_categorical("act", ["relu","leaky_relu","elu","tanh","selu"])
        drop= trial.suggest_float("dropout",0.0,0.5)
        
        opt_name = trial.suggest_categorical("optimizer", 
            ["adamw", "adam", "sgd", "rmsprop", "nadam", "radam", "adamax", "adagrad"])
        
        if opt_name == "sgd":
            # SGD needs higher learning rates
            lr = trial.suggest_float("lr", 1e-2, 1e-1, log=True)  # Changed minimum from 1e-4 to 1e-2
        else:
            # Adaptive optimizers work well with this range
            lr = trial.suggest_float("lr", 1e-4, 3e-2, log=True)
        
        batch=trial.suggest_int("batch", self.batch_range[0], self.batch_range[1], log=True)
        
        # loss function selection for TRAINING only
        loss_name = trial.suggest_categorical("loss_fn", ["mse", "mae", "huber", "smooth_l1", "log_cosh"])

        train_dl,val_dl,_ = self._make_dloaders(batch)
        model = MLPQualityPredictor(self.input_dim, self.output_dim, hl, neu, act, drop).to(DEVICE)
        
        # Use selected loss function for TRAINING only, we want to keep the validation metric consistent for comparison
        crit_train = make_loss_function(loss_name)
        
        # Define a consistent validation metric across all trials, I chose MSELoss for this
        metric_val_fn = nn.MSELoss()  # Using MSE as the canonical validation metric
        
        opt = make_optimizer(opt_name, model.parameters(), lr)

        best_val=float("inf"); best_ep=0
        for epoch in range(250): #epochs for training
            # ----- TRAINING -----
            model.train()
            for b in train_dl: #b is a batch of data
                #the input is the latent vector + process parameters together
                x=b["input"].to(DEVICE); y=b["target"].to(DEVICE)
                opt.zero_grad()
                # Use training-specific loss for backprop
                loss=crit_train(model(x),y)
                loss.backward()
                opt.step()
                
            # ----- VALIDATION -----
            model.eval(); vl=[]
            with torch.no_grad():
                for b in val_dl:
                    x=b["input"].to(DEVICE); y=b["target"].to(DEVICE)
                    # Use consistent validation metric for evaluation
                    vl.append(metric_val_fn(model(x),y).item())
                    
            v=float(np.mean(vl))
            trial.report(v, epoch)
            if trial.should_prune(): raise optuna.TrialPruned()
            if v<best_val: best_val,best_ep=v,epoch
            elif epoch-best_ep>25: break
            
        return best_val

    def _save_study_results(self, study):
        """Save detailed information about all trials to a text file."""
        out_path = self.output_dir / f"study_summary_{self.study_name}.txt"
        
        with open(out_path, "w") as f:
            # Write header with general study info
            f.write(f"Study Summary: {self.study_name}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Best Trial: {study.best_trial.number}\n")
            f.write(f"Best Value: {study.best_value:.6f}\n\n")
            
            # Write best parameters
            f.write("Best Parameters:\n")
            params = study.best_params
            
            # Format hidden layers
            neurons = [params[f"n{i}"] for i in range(params["hidden_layers"])]
            f.write(f"  Architecture: {params['hidden_layers']} layers with {neurons} neurons\n")
            f.write(f"  Activation: {params['act']}\n")
            f.write(f"  Dropout: {params['dropout']:.4f}\n")
            f.write(f"  Batch Size: {params['batch']}\n")
            f.write(f"  Optimizer: {params['optimizer']}\n")
            f.write(f"  Learning Rate: {params['lr']:.6f}\n")
            f.write(f"  Loss Function: {params['loss_fn']}\n\n")
            
            # Write trial summaries in table format
            f.write("-"*80 + "\n")
            f.write(f"{'Trial':^6} | {'Value':^12} | {'Hidden':^6} | {'Activ':^8} | {'Dropout':^8} | {'LR':^10} | Loss\n")
            f.write("-"*80 + "\n")
            
            for trial in sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf')):
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    try:
                        p = trial.params
                        hl = p["hidden_layers"]
                        neurons_str = str([p[f"n{i}"] for i in range(hl)])
                        f.write(f"{trial.number:^6} | {trial.value:^12.6f} | {hl:^6} | {p['act']:^8} | "
                                f"{p['dropout']:^8.3f} | {p['lr']:^10.6f} | {p['loss_fn']}\n")
                    except KeyError:
                        # Handle incomplete trials
                        f.write(f"{trial.number:^6} | {trial.value:^12.6f} | (incomplete parameters)\n")
            
            f.write("-"*80 + "\n")
        
        logger.info(f"Full study results saved to {out_path}")

    def _save_test_results_csv(self, train_loss, eval_loss, pred_values, true_values, loss_fn_name):
        
        # Save summary metrics
        metrics_path = self.output_dir / f"test_metrics_{self.study_name}.csv"
        with open(metrics_path, "w") as f:
            f.write("Metric,Value\n")
            f.write(f"Test_{loss_fn_name},{train_loss:.6f}\n")
            f.write(f"Test_MSE,{eval_loss:.6f}\n")
            
            # Calculate per-target MSE
            for i, col in enumerate(self.tgt_cols):
                target_mse = ((pred_values[:, i] - true_values[:, i])**2).mean()
                f.write(f"MSE_{col},{target_mse:.6f}\n")
                
                # Add R² coefficient determination
                r2 = 1.0 - (((pred_values[:, i] - true_values[:, i])**2).sum() / 
                           ((true_values[:, i] - true_values[:, i].mean())**2).sum())
                f.write(f"R2_{col},{r2:.6f}\n")
                
        # Save predictions vs actual values
        predictions_path = self.output_dir / f"test_predictions_{self.study_name}.csv"
        
        # Create header row
        header = []
        for col in self.tgt_cols:
            header.extend([f"Pred_{col}", f"True_{col}", f"Error_{col}"])
        
        with open(predictions_path, "w") as f:
            f.write(",".join(header) + "\n")
            
            # Write each row of predictions
            for i in range(len(pred_values)):
                row = []
                for j in range(len(self.tgt_cols)):
                    # Predicted value, true value, and error
                    pred = pred_values[i, j]
                    true = true_values[i, j]
                    error = pred - true
                    row.extend([f"{pred:.6f}", f"{true:.6f}", f"{error:.6f}"])
                f.write(",".join(row) + "\n")
                
        logger.info(f"Test metrics saved to {metrics_path}")
        logger.info(f"Test predictions saved to {predictions_path}")

    def optimize(self):
        study = optuna.create_study(
            direction="minimize", study_name=self.study_name,
            storage=self.storage, load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        )
        # After optimization is complete
        study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=True)
        logger.info(f"Best loss (val) : {study.best_value:.5f}")
        logger.info(f"Best params     : {study.best_params}")
        
        # ---------- Final Training With The Best Parameters -------------------------------
        p=study.best_params
        neu=[p[f"n{i}"] for i in range(p["hidden_layers"])]
        model=MLPQualityPredictor(self.input_dim,self.output_dim,
                                  p["hidden_layers"],neu,
                                  p["act"],p["dropout"]).to(DEVICE)
        batch=p["batch"]
        tv_df=pd.concat([self.train_df,self.val_df],ignore_index=True)

        # Re-fit scalers on the combined train+val data
        self.scaler_proc = StandardScaler().fit(
            tv_df[self.proc_cols].values.astype(float))
        self.scaler_tgt = StandardScaler().fit(
            tv_df[self.tgt_cols].values.astype(float))

        # Avoid div-by-zero for constant cols
        self.scaler_proc.scale_[self.scaler_proc.scale_ == 0] = 1.0
        self.scaler_tgt.scale_[self.scaler_tgt.scale_ == 0] = 1.0

        # Update class attributes
        self.train_df = tv_df
        self.val_df = pd.DataFrame([])

        # Now create data loaders
        train_dl, _, test_dl = self._make_dloaders(batch)

        # Use best training loss function from optimization
        crit_train = make_loss_function(p["loss_fn"])
        
        # But also use the same validation metric as during optimization
        metric_val_fn = nn.MSELoss()
        
        opt = make_optimizer(p["optimizer"], model.parameters(), p["lr"])

        for _ in range(150): # epochs for final training
            model.train()
            for b in train_dl:
                x=b["input"].to(DEVICE); y=b["target"].to(DEVICE)
                opt.zero_grad(); crit_train(model(x),y).backward(); opt.step()

        # Test using BOTH metrics for completeness
        model.eval()
        test_train_loss = []  # Using the training loss on test set
        test_eval_loss = []   # Using the validation metric on test set
        
        # Store predictions and ground truth for CSV export
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for b in test_dl:
                x=b["input"].to(DEVICE); y=b["target"].to(DEVICE)
                pred = model(x)
                test_train_loss.append(crit_train(pred, y).item())
                test_eval_loss.append(metric_val_fn(pred, y).item())
                
                # Convert scaled predictions and targets back to original units
                pred_phys = inverse_transform_tgt(pred, self.scaler_tgt)
                y_phys = inverse_transform_tgt(y, self.scaler_tgt)
                
                # Store for CSV export
                all_predictions.append(pred_phys.cpu().numpy())
                all_targets.append(y_phys.cpu().numpy())
        
        test_train_loss_avg = np.mean(test_train_loss)
        test_eval_loss_avg = np.mean(test_eval_loss)
        
        logger.info(f"Test loss (training criterion {p['loss_fn']}): {test_train_loss_avg:.5f}")
        logger.info(f"Test loss (validation metric MSE): {test_eval_loss_avg:.5f}")
        
        # Export test results to CSV
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        self._save_test_results_csv(test_train_loss_avg, test_eval_loss_avg, 
                                   all_predictions, all_targets, p["loss_fn"])
        
        self._save_study_results(study)
        
        # ---------- save
        out = self.output_dir / f"best_mlp_{self.study_name}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "hyperparams": {**p, "input_dim": self.input_dim, "output_dim": self.output_dim},
            "proc_mean": self.scaler_proc.mean_,
            "proc_scale": self.scaler_proc.scale_,
            "tgt_mean": self.scaler_tgt.mean_,
            "tgt_scale": self.scaler_tgt.scale_,
            "loss_fn": p["loss_fn"],
            "optimizer": p["optimizer"],
        }, out)
        logger.info(f"Saved best model → {out.resolve()}")
        
        return model, study

def make_optimizer(name: str, model_params, lr: float):
    """Create different optimizers based on name."""
    if name == "adamw":
        return optim.AdamW(model_params, lr=lr)
    elif name == "adam":
        return optim.Adam(model_params, lr=lr)
    elif name == "sgd":
        return optim.SGD(model_params, lr=lr, momentum=0.9)
    elif name == "rmsprop":
        return optim.RMSprop(model_params, lr=lr)
    elif name == "nadam":
        return optim.NAdam(model_params, lr=lr)
    elif name == "radam":
        return optim.RAdam(model_params, lr=lr)
    elif name == "adamax":
        return optim.Adamax(model_params, lr=lr)
    elif name == "adagrad":
        return optim.Adagrad(model_params, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

# ────────────────────────────────────────────────────────────────────────────
# Helper: convert network output back to physical units, so we can compare to targets
# ────────────────────────────────────────────────────────────────────────────
def inverse_transform_tgt(arr, scaler_tgt: StandardScaler):
    """
    Convert a numpy or Torch tensor that lives in *scaled target space*
    back to the original engineering units using `scaler_tgt`.
    """
    was_torch = torch.is_tensor(arr)
    if was_torch:
        cpu_arr = arr.detach().cpu().numpy()
    else:
        cpu_arr = arr
    orig = scaler_tgt.inverse_transform(cpu_arr).astype(np.float32)   # keep float32
    return torch.from_numpy(orig).to(arr.device) if was_torch else orig


PROC_COLS = (
    "Packing Pressure", 
    "Mold Surface Temperature",
    "Melt Temperature",
    "Injection Time",
    "gate location to furthest point in x",
    "gate location to furthest point in y",
    "gate location to furthest point in z",
    "gate location to COM",
)
TGT_COLS = ("CavityWeight", "MaxWarp", "VolumetricShrinkage")

# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ' Thesis system is the direct latent space implementation, hopefully this is clear'
    MAIN_FOLDER = r"C:\Users\maart\OneDrive - KU Leuven\KUL\MOAI\Master thesis\code\SYSTEM\TrainingData_Thesis_System"
    AUTOENCODER_PATH = r"C:\Users\maart\OneDrive - KU Leuven\KUL\MOAI\Master thesis\code\SYSTEM\autoencoder_best.pt"
    OUTPUT_DIR = "Thesis_System_Results"  
     
    trainer = MLPTrainer(
        main_folder = MAIN_FOLDER,
        autoencoder_path = AUTOENCODER_PATH,
        latent_dim = 256,
        proc_cols = PROC_COLS,
        tgt_cols = TGT_COLS,
        batch_range = (8,128),
        n_trials = 5000,
        study_name = "mlp_quality_prediction",
        storage = None,
        output_dir = OUTPUT_DIR, 
    )
    best_model, study = trainer.optimize()
