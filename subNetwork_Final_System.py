"""
Hierarchical-MLP quality-prediction training pipeline
=====================================================
•   <product>/                              
        ├─ <product>.xlsx  (80 rows × 8 process cols + 3 target cols)
        └─ <anything>.npy  (voxel grid)

The model now consists of **three** MLPs:
    ▸ MLP_lat   : maps the AE latent vector  →  R^h_lat
    ▸ MLP_proc  : maps the 8 process params  →  R^h_proc
    ▸ MLP_fuse  : maps [h_lat ⨁ h_proc]      →  3 quality metrics
Splitting is still **by product** (no leakage).
"""
# ────────────────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────────────────
import os, logging, random, math, json, time, requests
from pathlib import Path

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import optuna
from optuna.trial import Trial
import warnings
from sklearn.exceptions import DataConversionWarning

# Suppress specific warnings (annoying ones)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")
# ────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# ────────────────────────────────────────────────────────────────────────────
# 3-D auto-encoder  (entire architecture)
# ────────────────────────────────────────────────────────────────────────────
class FinalAutoencoder3D(nn.Module):
    """
    Autoencoder for 128^3 volumes.
    - kernel_size=3, padding=1, stride=2 in each downsample block => 128->64->32->16->8->4
    - device attribute is stored automatically
    - latent_dim is a hyperparam for tuning
    """
    def __init__(self, latent_dim=512):
        super().__init__()
        self._device = torch.device("cpu")

        # Hardcoded kernel=3, pad=1, stride=2
        kernel_size = 3
        padding = 1

        # -- Encoder (4 downsampling blocks) --
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=kernel_size, stride=2, padding=padding),
            nn.PReLU(),
            nn.Conv3d(16, 16, kernel_size=kernel_size, stride=1, padding=padding),
            nn.PReLU(),

            nn.Conv3d(16, 32, kernel_size=kernel_size, stride=2, padding=padding),
            nn.PReLU(),
            nn.Conv3d(32, 32, kernel_size=kernel_size, stride=1, padding=padding),
            nn.PReLU(),

            nn.Conv3d(32, 64, kernel_size=kernel_size, stride=2, padding=padding),
            nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=kernel_size, stride=1, padding=padding),
            nn.PReLU(),

            nn.Conv3d(64, 128, kernel_size=kernel_size, stride=2, padding=padding),
            nn.PReLU(),
            nn.Conv3d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.PReLU(),
        )

        # Flatten => fc => latent
        self.fc_mu = nn.Linear(128 * 8 * 8 * 8, latent_dim)
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
            nn.Conv3d(4, 1, kernel_size=1),
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

def load_autoencoder(path:str, device, latent_dim:int=256):
    chk = torch.load(path, map_location=device)
    sd  = chk["model_state_dict"] if "model_state_dict" in chk else chk
    
    # Filter to ONLY include encoder parameters and ignore decoder completely 
    encoder_sd = {k: v for k, v in sd.items() if k.startswith('encoder.') or k.startswith('fc_mu.')}
    
    model = FinalAutoencoder3D(latent_dim).to(device)
    model.load_state_dict(encoder_sd, strict=False)
    model.eval()
    logger.info(f"Loaded AE encoder from {path} ({len(encoder_sd)} parameters)")
    return model

# ────────────────────────────────────────────────────────────────────────────
# DATASET
# ────────────────────────────────────────────────────────────────────────────
class InjectionMoldingDataset(Dataset):
    """
    returns dict(lat   = latent-vector      (torch.float32)
                 proc  = 8-D process vector (torch.float32)
                 target= 3-D target         (torch.float32, scaled) )
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

        # process parameters
        Xp = self.df[self.proc_cols].values.astype(np.float32)
        self.proc_arr = scaler_proc.transform(Xp) if scaler_proc else Xp

        # targets
        yt = self.df[self.tgt_cols].values.astype(np.float32)
        self.tgt_arr  = scaler_tgt.transform(yt) if scaler_tgt else yt

        # cache latents
        self.latent_cache={}
        with torch.no_grad():
            for _,row in self.df[["product_name","cad_model_file"]].drop_duplicates().iterrows():
                vox = np.load(voxel_root/row["product_name"]/row["cad_model_file"]).astype(np.float32)[None,None]
                lat = self.ae.encode(torch.from_numpy(vox).to(device))
                self.latent_cache[row["cad_model_file"]] = lat.cpu().numpy().squeeze()

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        return {
            "lat"   : torch.tensor(self.latent_cache[row["cad_model_file"]]),
            "proc"  : torch.tensor(self.proc_arr[idx]),
            "target": torch.tensor(self.tgt_arr [idx]),
        }

# ────────────────────────────────────────────────────────────────────────────
# Model – three-tower MLP
# ────────────────────────────────────────────────────────────────────────────
def activation(name): return {
    "relu":nn.ReLU(), "leaky_relu":nn.LeakyReLU(),
    "elu":nn.ELU(),   "tanh":nn.Tanh(), "selu":nn.SELU()
}[name]

def multilayer(dim_in:int, hidden:list[int], act_name:str, dropout:float):
    act = activation(act_name); layers=[]
    for h in hidden:
        layers += [nn.Linear(dim_in,h), act]
        if dropout>0: layers.append(nn.Dropout(dropout))
        dim_in = h
    return nn.Sequential(*layers), dim_in

class HierarchicalQualityPredictor(nn.Module):
    """
    • MLP_lat   : latent → h_lat
    • MLP_proc  : proc   → h_proc
    • MLP_fuse  : concat → 3
    """
    def __init__(self,
             latent_dim:int, proc_dim:int,
             lat_hidden:list[int], proc_hidden:list[int], fuse_hidden:list[int], 
             lat_act:str="relu", proc_act:str="relu", fuse_act:str="relu",
             lat_drop:float=0.0, proc_drop:float=0.0, fuse_drop:float=0.0):
        super().__init__()
        self.lat_tower , h_lat  = multilayer(latent_dim, lat_hidden, lat_act, lat_drop)
        self.proc_tower, h_proc = multilayer(proc_dim, proc_hidden, proc_act, proc_drop)
        self.fuse_net , _       = multilayer(h_lat+h_proc, fuse_hidden, fuse_act, fuse_drop)
        self.out = nn.Linear(fuse_hidden[-1] if fuse_hidden else h_lat+h_proc, 3)

    def forward(self, lat, proc):
        z_lat  = self.lat_tower(lat)
        z_proc = self.proc_tower(proc)
        fused  = torch.cat([z_lat, z_proc], dim=1)
        return self.out(self.fuse_net(fused))

# ────────────────────────────────────────────────────────────────────────────
# Helper – loss & optim
# ────────────────────────────────────────────────────────────────────────────
def make_loss(name):
    if name=="mse": return nn.MSELoss()
    if name=="mae": return nn.L1Loss()
    if name=="huber": return nn.HuberLoss(delta=1.0)
    if name=="smooth_l1": return nn.SmoothL1Loss(beta=1.0)
    if name=="log_cosh":
        return lambda y_hat, y: torch.mean(torch.log(torch.cosh(y_hat-y)))
    raise ValueError(name)

def make_optimizer(name, params, lr):
    if name=="adamw":  return optim.AdamW(params, lr=lr)
    if name=="adam":   return optim.Adam (params, lr=lr)
    if name=="sgd":    return optim.SGD  (params, lr=lr)
    if name=="rmsprop":return optim.RMSprop(params, lr=lr)
    if name=="nadam":  return optim.NAdam(params, lr=lr)
    if name=="radam":  return optim.RAdam(params, lr=lr)
    if name=="adamax": return optim.Adamax(params, lr=lr)
    if name=="adagrad":return optim.Adagrad(params, lr=lr)
    raise ValueError(name)

# ────────────────────────────────────────────────────────────────────────────
# Trainer
# ────────────────────────────────────────────────────────────────────────────
class Trainer:
    def __init__(self,
        main_folder, autoencoder_path, latent_dim,
        proc_cols, tgt_cols,
        batch_range=(8,64), val_ratio=.1, test_ratio=.2, random_state=42,
        n_trials=200, study_name="hier_mlp", storage=None,
        output_dir="Subnetwork_Results",  # Add output directory parameter
    ):
        self.main = Path(main_folder)
        self.proc_cols, self.tgt_cols = list(proc_cols), list(tgt_cols)
        self.batch_rng, self.val_r, self.test_r = batch_range, val_ratio, test_ratio
        self.n_trials, self.study_name, self.storage = n_trials, study_name, storage
        self.latent_dim = latent_dim
        
        # Create output directory for results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Results will be saved to {self.output_dir.resolve()}")

        self.ae  = load_autoencoder(autoencoder_path, DEVICE, latent_dim)
        self.df  = self._load_df()
        self.train_df, self.val_df, self.test_df = self._split()

        self.scaler_proc = StandardScaler().fit(self.train_df[self.proc_cols])
        self.scaler_tgt  = StandardScaler().fit(self.train_df[self.tgt_cols])
        for sc in (self.scaler_proc, self.scaler_tgt):
            sc.scale_[sc.scale_==0]=1.0

        self.trials_done, self.best_val = 0, float("inf")

    # ---------- data -------------------------------------------------------
    def _load_df(self):
        rows=[]
        for prod_dir in self.main.iterdir():
            if not prod_dir.is_dir(): continue
            try:
                xl = next(f for f in prod_dir.iterdir() if f.suffix in (".xlsx",".xls"))
                npy= next(f for f in prod_dir.iterdir() if f.suffix==".npy")
            except StopIteration: continue
            df = pd.read_excel(xl)
            df["product_name"]=prod_dir.name; df["cad_model_file"]=npy.name
            rows.append(df)
        if not rows: raise RuntimeError("No products found")
        return pd.concat(rows, ignore_index=True)

    def _split(self):
        """Split data using canonical product split from JSON file"""
        # Load the canonical product split
        with open("canonical_product_split.json", "r") as f:
            product_split = json.load(f)
        
        # Log which products are in each split
        logger.info("Using canonical product split:")
        logger.info(f"Train products: {len(product_split['train_products'])} - {product_split['train_products']}")
        logger.info(f"Val products: {len(product_split['val_products'])} - {product_split['val_products']}")
        logger.info(f"Test products: {len(product_split['test_products'])} - {product_split['test_products']}")
        
        # Filter data based on product split
        train_df = self.df[self.df["product_name"].isin(product_split["train_products"])].reset_index(drop=True)
        val_df = self.df[self.df["product_name"].isin(product_split["val_products"])].reset_index(drop=True)
        test_df = self.df[self.df["product_name"].isin(product_split["test_products"])].reset_index(drop=True)
        
        # Log split sizes
        logger.info(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df

    def _dl(self, df, batch, shuffle=False):
        ds=InjectionMoldingDataset(df,self.main,self.ae,
                                   self.proc_cols,self.tgt_cols,
                                   self.scaler_proc,self.scaler_tgt,DEVICE)
        return DataLoader(ds,batch,shuffle=shuffle,num_workers=0,pin_memory=True)

    # ---------- objective ---------------------------------------------------
    def _objective(self, trial:Trial):
        # large range for neurons
        lat_h = [trial.suggest_int(f"lat_h{i}",16,512,log=True)
                 for i in range(trial.suggest_int("lat_layers",1,10))]
        proc_h = [trial.suggest_int(f"prc_h{i}",16,512,log=True)
                  for i in range(trial.suggest_int("proc_layers",1,10))]
        fuse_h = [trial.suggest_int(f"fus_h{i}",16,512,log=True)
                  for i in range(trial.suggest_int("fuse_layers",1,10))]
        
        # Activation and dropout parameters 
        lat_act = trial.suggest_categorical("lat_act", ["relu","leaky_relu","elu","tanh","selu"])
        proc_act = trial.suggest_categorical("proc_act", ["relu","leaky_relu","elu","tanh","selu"])
        fuse_act = trial.suggest_categorical("fuse_act", ["relu","leaky_relu","elu","tanh","selu"])
        
        lat_drop = trial.suggest_float("lat_drop", 0.0, 0.5)
        proc_drop = trial.suggest_float("proc_drop", 0.0, 0.5)
        fuse_drop = trial.suggest_float("fuse_drop", 0.0, 0.5)
        
        batch = trial.suggest_int("batch", self.batch_rng[0], self.batch_rng[1], log=True)
        opt_n = trial.suggest_categorical("opt", 
                ["adamw", "adam", "sgd", "rmsprop", "nadam", "radam", "adamax", "adagrad"])
        
        # Learning rate depends on SGD or not because SGD is more sensitive
        if opt_n == "sgd":
            lr = trial.suggest_float("lr", 1e-2, 1e-0, log=True)
        else:
            lr = trial.suggest_float("lr", 1e-4, 3e-2, log=True)

        # This loss is only for training, we use a fixed metric for validation for good comparison
        loss_n = trial.suggest_categorical("loss", ["mse","mae","huber","smooth_l1","log_cosh"])
        
        train_dl = self._dl(self.train_df, batch, shuffle=True)
        val_dl = self._dl(self.val_df, batch, shuffle=False)

        model = HierarchicalQualityPredictor(
            self.latent_dim, len(self.proc_cols),
            lat_h, proc_h, fuse_h,
            lat_act, proc_act, fuse_act,
            lat_drop, proc_drop, fuse_drop).to(DEVICE)

        crit_train = make_loss(loss_n)  # Trial-specific loss for training
        fixed_metric = nn.MSELoss()     # Fixed metric for validation
        
        opt = make_optimizer(opt_n, model.parameters(), lr)

        best = float("inf"); patience = 25; last_best = 0
        for ep in range(250):
            # Training with trial-specific loss
            model.train()
            for b in train_dl:
                opt.zero_grad()
                pred = model(b["lat"].to(DEVICE), b["proc"].to(DEVICE))
                loss = crit_train(pred, b["target"].to(DEVICE))
                loss.backward()
                opt.step()
                
            # Validation with FIXED metric
            model.eval(); v = []
            with torch.no_grad():
                for b in val_dl:
                    pred = model(b["lat"].to(DEVICE), b["proc"].to(DEVICE))
                    # Use fixed metric for validation
                    v.append(fixed_metric(pred, b["target"].to(DEVICE)).item())
                    
            val = np.mean(v)
            trial.report(val, ep)
            if trial.should_prune(): raise optuna.TrialPruned()
            if val < best: best, last_best = val, ep
            elif ep - last_best > patience: break

        self.trials_done += 1
        return best  # Return the fixed metric value

    # -----------------------------------------------------------------------
    def optimize(self):
        # Study creation Optuna
        study = optuna.create_study(direction="minimize", study_name=self.study_name,
                              storage=self.storage, load_if_exists=True,
                              pruner=optuna.pruners.MedianPruner(10))
        study.optimize(self._objective, n_trials=self.n_trials,
                       show_progress_bar=True)

        p = study.best_params
        lat_h = [p[f"lat_h{i}"] for i in range(p["lat_layers"])]
        proc_h = [p[f"prc_h{i}"] for i in range(p["proc_layers"])]
        fuse_h = [p[f"fus_h{i}"] for i in range(p["fuse_layers"])]
        batch = p["batch"]

        model = HierarchicalQualityPredictor(
            self.latent_dim, len(self.proc_cols),
            lat_h, proc_h, fuse_h,
            p["lat_act"], p["proc_act"], p["fuse_act"],
            p["lat_drop"], p["proc_drop"], p["fuse_drop"]).to(DEVICE)
        
        # Training uses the best loss function from the study
        crit_train = make_loss(p["loss"])
        
        # But we keep the fixed metric for evaluation
        fixed_metric = nn.MSELoss()
        
        opt = make_optimizer(p["opt"], model.parameters(), p["lr"])

        # Merge train+val datasets
        full_tr = pd.concat([self.train_df, self.val_df], ignore_index=True)
        
        # Create dataloaders
        train_dl = self._dl(full_tr, batch, shuffle=True)
        test_dl = self._dl(self.test_df, batch, shuffle=False)

        # Final training
        for epoch in range(150):
            model.train()
            for b in train_dl:
                opt.zero_grad()
                pred = model(b["lat"].to(DEVICE), b["proc"].to(DEVICE))
                loss = crit_train(pred, b["target"].to(DEVICE))
                loss.backward()
                opt.step()

        # Test evaluation
        model.eval(); test_losses = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for b in test_dl:
                pred = model(b["lat"].to(DEVICE), b["proc"].to(DEVICE))
                test_losses.append(fixed_metric(pred, b["target"].to(DEVICE)).item())
                
                # Store predictions and targets for analysis
                all_predictions.append(pred.cpu().numpy())
                all_targets.append(b["target"].cpu().numpy())
        
        test_loss = np.mean(test_losses)
        logger.info(f"Test loss: {test_loss:.6f}")
        
        # Convert lists of batches to full arrays
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        # Save test results to CSV
        self._save_test_results_csv(test_loss, all_predictions, all_targets)
        
        # Save study results to text file
        self._save_study_results(study)
        
        # Save model
        model_path = self.output_dir / f"best_hier_mlp_{self.study_name}.pt"
        torch.save(
            {"state_dict": model.state_dict(), "params": p,
             "proc_mean": self.scaler_proc.mean_, "proc_scale": self.scaler_proc.scale_,
             "tgt_mean": self.scaler_tgt.mean_, "tgt_scale": self.scaler_tgt.scale_},
            model_path)
        logger.info(f"Saved best model → {model_path}")
        
        return model, study

    # Then add the _save_study_results method to save trial info
    def _save_study_results(self, study):
        """Save detailed information about all trials to a text file."""
        out_path = self.output_dir / f"study_summary_{self.study_name}.txt"
        
        with open(out_path, "w") as f:
            # Write header
            f.write(f"Hierarchical Model Study Summary: {self.study_name}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Best Trial: {study.best_trial.number}\n")
            f.write(f"Best Value: {study.best_value:.6f}\n\n")
            
            # Format parameters
            p = study.best_params
            lat_h = [p[f"lat_h{i}"] for i in range(p["lat_layers"])]
            proc_h = [p[f"prc_h{i}"] for i in range(p["proc_layers"])]
            fuse_h = [p[f"fus_h{i}"] for i in range(p["fuse_layers"])]
            
            f.write("Best Architecture:\n")
            f.write(f"  Latent tower: {lat_h}\n")
            f.write(f"  Process tower: {proc_h}\n")
            f.write(f"  Fusion tower: {fuse_h}\n\n")
            
            f.write(f"  Latent activation: {p['lat_act']}\n")
            f.write(f"  Process activation: {p['proc_act']}\n")
            f.write(f"  Fusion activation: {p['fuse_act']}\n\n")
            
            f.write(f"  Latent dropout: {p['lat_drop']:.4f}\n")
            f.write(f"  Process dropout: {p['proc_drop']:.4f}\n")
            f.write(f"  Fusion dropout: {p['fuse_drop']:.4f}\n\n")
            
            f.write(f"  Batch size: {p['batch']}\n")
            f.write(f"  Optimizer: {p['opt']}\n")
            f.write(f"  Learning rate: {p['lr']:.6f}\n")
            f.write(f"  Loss function: {p['loss']}\n\n")
            
            # All trials table
            f.write("-"*80 + "\n")
            f.write(f"{'Trial':^6} | {'Value':^12} | {'Latent':^12} | {'Process':^12} | {'Fusion':^12} | {'Loss':^8}\n")
            f.write("-"*80 + "\n")
            
            for trial in sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf')):
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    try:
                        p = trial.params
                        lat_layers = p["lat_layers"]
                        proc_layers = p["proc_layers"]
                        fuse_layers = p["fuse_layers"]
                        
                        # Just show layer counts in table for brevity
                        f.write(f"{trial.number:^6} | {trial.value:^12.6f} | "
                                f"{lat_layers:^12d} | {proc_layers:^12d} | "
                                f"{fuse_layers:^12d} | {p['loss']:^8}\n")
                    except KeyError:
                        # Handle incomplete trials
                        f.write(f"{trial.number:^6} | {trial.value:^12.6f} | (incomplete parameters)\n")
        
        logger.info(f"Full study results saved to {out_path}")

    # Add the _save_test_results method
    def _save_test_results_csv(self, test_loss, predictions, targets):
        """Save test results to CSV files"""
        # Save summary metrics
        metrics_path = self.output_dir / f"test_metrics_{self.study_name}.csv"
        with open(metrics_path, "w") as f:
            f.write("Metric,Value\n")
            f.write(f"Test_MSE_Overall,{test_loss:.6f}\n")
            
            # Calculate per-target MSE
            for i, col in enumerate(self.tgt_cols):
                target_mse = ((predictions[:, i] - targets[:, i])**2).mean()
                f.write(f"MSE_{col},{target_mse:.6f}\n")
                
                # Add R² coefficient of determination
                r2 = 1.0 - (((predictions[:, i] - targets[:, i])**2).sum() / 
                           ((targets[:, i] - targets[:, i].mean())**2).sum())
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
            for i in range(len(predictions)):
                row = []
                for j in range(len(self.tgt_cols)):
                    # Predicted value, true value, and error
                    pred = predictions[i, j]
                    true = targets[i, j]
                    error = pred - true
                    row.extend([f"{pred:.6f}", f"{true:.6f}", f"{error:.6f}"])
                f.write(",".join(row) + "\n")
                
        logger.info(f"Test metrics saved to {metrics_path}")
        logger.info(f"Test predictions saved to {predictions_path}")

# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
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

if __name__ == "__main__":
    MAIN_FOLDER = r"C:\Users\maart\OneDrive - KU Leuven\KUL\MOAI\Master thesis\code\SYSTEM\TrainingData_Thesis_System"
    AUTOENCODER_PATH = r"C:\Users\maart\OneDrive - KU Leuven\KUL\MOAI\Master thesis\code\SYSTEM\autoencoder_best.pt"
    OUTPUT_DIR = r"C:\Users\maart\OneDrive - KU Leuven\KUL\MOAI\Master thesis\code\SYSTEM\Subnetwork_Results"
    
    trainer = Trainer(
        MAIN_FOLDER, AUTOENCODER_PATH, latent_dim=256,
        proc_cols=PROC_COLS, tgt_cols=TGT_COLS,
        batch_range=(8,128), n_trials=5000,
        output_dir=OUTPUT_DIR
    )
    trainer.optimize()
