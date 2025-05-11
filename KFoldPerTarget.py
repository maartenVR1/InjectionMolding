from __future__ import annotations
# kfold_all_systems.py
"""
Grouped k‑fold cross‑validation for ALL three network types used in the thesis:
--------------------------------------------------------------------------
1. **Benchmark System** – classic MLP on 12 geometric + process features
2. **Sub‑Network System** – hierarchical 3‑tower network (latent / process / fusion)
3. **Thesis / Normal System** – latent vector concatenated with 8 process parameters (of which 4 are gate location related)

For every product the rows belong to the same fold (GroupKFold on "product_name").
The exact (already optimised) architectures + hyper‑parameters are hard‑coded below.
Adjust only `MAIN_FOLDER_*` and (optionally) `AE_WEIGHTS` to fit your directory layout.

Usage
-----
python kfold_all_systems.py

after installing the Python packages listed at the top of the file.
The script prints the mean ± std validation MSE for each system.
"""


""""
I think the SGD gradient is exploding in the benchmark system. 
This would mean that the model weights are not being updated correctly, due to the exploding gradient problem, which causes them to diverge.
Gradient explosion occurs when the gradients computed during backpropagation become extremely large or even approach infinity. This happens because:

Compounding Effect: During backpropagation, gradients are multiplied through the network layers according to the chain rule. If some of these gradients are >1, their repeated multiplication can cause values to grow exponentially.

Cascading Effect: As one layer's gradients explode, they make the previous layer's gradients even larger, creating a cascading effect.

Mathematical Result: When the loss contains very large values, its derivatives (gradients) can also become excessively large.

Symptoms:

Loss suddenly jumping to inf
Model weights growing to extreme values
Training becoming unstable after initially working fine

"""

import os
import json
from pathlib import Path
from typing import List, Tuple
import logging
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

# Configuration for repeated cross-validation
N_REPEATS = 50 

def set_training_seeds(seed):
    #randomness for reproducibility
    # Set the seed for all random number generators to ensure reproducibility
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --------------------------------------------------------------------------
# Logging & device
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# --------------------------------------------------------------------------
# 1 ─ BENCHMARK SYSTEM  ─────────────────────────────────────────────────────
# --------------------------------------------------------------------------
from Benchmark_System import (
    BenchmarkDataset,
    MLPBenchmarkModel,
    BenchmarkTrainer,  # only to reuse its _load_multi_folder_data helper
)

BEST_BENCHMARK = dict(
    batch_size=10,
    n_layers=1,
    n_units_l0=422,
    activation="elu",
    dropout=0.08816565007014701,
    optimizer="SGD",
    lr=0.026160823416352806,
    loss="MSE",
)
# folders / columns
MAIN_FOLDER_BENCH = Path(r"C:\Users\maart\OneDrive - KU Leuven\KUL\MOAI\Master thesis\code\SYSTEM\TrainingData_Benchmark_System")
INPUT_COLS_BENCH = [
    'Cavity Volume (mm3)', 'Cavity Surface Area (mm2)',
    'Projection Area zx (mm2)', 'Projection Area yz (mm2)',
    'gate location to furthest point in x', 'gate location to furthest point in y',
    'gate location to furthest point in z', 'gate location to COM',
    'Mold Surface Temperature', 'Melt Temperature',
    'Packing Pressure', 'Injection Time'
]
TARGET_COLS = ['CavityWeight', 'MaxWarp', 'VolumetricShrinkage']

# --------------------------------------------------------------------------
# 2 ─ SUB‑NETWORK (Hierarchical) SYSTEM  ────────────────────────────────────
# --------------------------------------------------------------------------
from subNetwork_Final_System import (
    InjectionMoldingDataset as SubnetDataset,
    HierarchicalQualityPredictor,
    load_autoencoder as load_ae_subnet,
)

SHARED_DATA_FOLDER = Path(r"C:\Users\maart\OneDrive - KU Leuven\KUL\MOAI\Master thesis\code\SYSTEM\TrainingData_Thesis_System")

MAIN_FOLDER_SUBNET = SHARED_DATA_FOLDER
AE_WEIGHTS_SUBNET = Path(r"C:\Users\maart\OneDrive - KU Leuven\KUL\MOAI\Master thesis\code\SYSTEM\autoencoder_best.pt")  # encoder‑only weights

# best architecture 
SUB_ARCH = dict(
    lat_hidden=[196, 215, 44, 414, 36, 240, 82, 127, 23, 215],
    proc_hidden=[36],
    fuse_hidden=[274],
    lat_act="tanh",
    proc_act="tanh",
    fuse_act="selu",
    lat_drop=0.4138,
    proc_drop=0.3247,
    fuse_drop=0.0948,
    batch_size=26,
    opt="nadam",
    lr=0.002188,
)

PROC_COLS = [
    "Packing Pressure", "Mold Surface Temperature", "Melt Temperature",
    "Injection Time",
    "gate location to furthest point in x", "gate location to furthest point in y",
    "gate location to furthest point in z", "gate location to COM",
]

# --------------------------------------------------------------------------
# 3 ─ THESIS / NORMAL SYSTEM  ───────────────────────────────────────────────
# --------------------------------------------------------------------------
from Thesis_System import (
    InjectionMoldingDataset as ThesisDataset,
    MLPQualityPredictor,
    load_autoencoder as load_ae_thesis,
    make_loss_function as make_loss_thesis,
    make_optimizer as make_opt_thesis,
)

MAIN_FOLDER_THESIS = SHARED_DATA_FOLDER
AE_WEIGHTS_THESIS = Path(r"C:\Users\maart\OneDrive - KU Leuven\KUL\MOAI\Master thesis\code\SYSTEM\autoencoder_best.pt")  # same encoder structure

THESIS_ARCH = dict(
    hidden_layers=3,
    neurons=[411, 163, 327],
    act="tanh",
    dropout=0.08465965112996494,
    batch_size=11,
    opt="radam",
    lr=0.01622181089749877,
    loss_fn="huber",
)

# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

def mse(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    return nn.MSELoss()(y_hat, y).item()

# Add this helper function to calculate MSE for each target separately
def target_specific_mse(y_hat: torch.Tensor, y: torch.Tensor) -> List[float]:
    """Calculate MSE for each target dimension separately."""
    mse_per_target = []
    for i in range(y.shape[1]):  # For each target dimension
        mse_per_target.append(nn.MSELoss()(y_hat[:, i], y[:, i]).item())
    return mse_per_target

# Add this helper function to calculate MSE in real units
def calculate_real_mse(y_hat: torch.Tensor, y: torch.Tensor, scaler) -> List[float]:
    """Calculate MSE for each target in real units (inverse transformed)"""
    # Convert to numpy and apply inverse transform
    y_hat_np = y_hat.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    
    # Inverse transform to get real units
    y_hat_real = scaler.inverse_transform(y_hat_np)
    y_real = scaler.inverse_transform(y_np)
    
    # Calculate MSE per target column
    real_mse_per_target = []
    for i in range(y_real.shape[1]):
        mse = ((y_hat_real[:, i] - y_real[:, i]) ** 2).mean()
        real_mse_per_target.append(mse)
    
    return real_mse_per_target

def grouped_kfold_indices(df: pd.DataFrame, n_splits: int = 5):
    gkf = GroupKFold(n_splits=n_splits)
    groups = df["product_name"].values
    for tr_idx, val_idx in gkf.split(df, groups=groups):
        yield tr_idx, val_idx


def load_product_dataframe(main_folder: Path) -> pd.DataFrame:
    """Load data from product folders into a single DataFrame with product metadata."""
    rows = []
    for prod in main_folder.iterdir():
        if not prod.is_dir(): continue
        try:
            xl = next(f for f in prod.iterdir() if f.suffix in (".xlsx", ".xls"))
            npy = next(f for f in prod.iterdir() if f.suffix == ".npy")
        except StopIteration:
            continue
        df = pd.read_excel(xl)
        df["product_name"] = prod.name
        df["cad_model_file"] = npy.name
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def write_results_to_file(filename, message):
    """Append message to the results file"""
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


# --------------------------------------------------------------------------
# 1. BENCHMARK Cross validaiton
# --------------------------------------------------------------------------

def run_benchmark_cv(k: int = 5):
    logger.info("\n=== BENCHMARK SYSTEM ===")
    tmp = BenchmarkTrainer(
        main_folder=str(MAIN_FOLDER_BENCH),
        input_columns=INPUT_COLS_BENCH,
        target_columns=TARGET_COLS,
        n_trials=1,
        output_dir="__tmp_bench_cv",
    )
    full_df = tmp._load_multi_folder_data()

    scores = []
    fold_scores = []
    # Create lists for each target
    target_scores = [[] for _ in range(len(TARGET_COLS))]
    real_target_scores = [[] for _ in range(len(TARGET_COLS))]
    
    for fold, (tr, va) in enumerate(grouped_kfold_indices(full_df, k), 1):
        tr_df, va_df = full_df.iloc[tr].reset_index(drop=True), full_df.iloc[va].reset_index(drop=True)
        
        # Print validation fold products
        val_products = sorted(va_df["product_name"].unique())
        logger.info(f"Fold {fold} validation products: {val_products}")

        x_scaler = StandardScaler().fit(tr_df[INPUT_COLS_BENCH])
        x_scaler.scale_[x_scaler.scale_ == 0] = 1.0
        y_scaler = StandardScaler().fit(tr_df[TARGET_COLS])
        y_scaler.scale_[y_scaler.scale_ == 0] = 1.0

        def make_ds(df, is_train):
            tmp_csv = f"__bench_fold{fold}_{'tr' if is_train else 'va'}.csv"
            df.to_csv(tmp_csv, index=False)
            ds = BenchmarkDataset(
                tmp_csv, 
                INPUT_COLS_BENCH, 
                TARGET_COLS, 
                transform=x_scaler,
                target_transform=y_scaler, 
                fit_transform=False
            )
            os.remove(tmp_csv)
            return ds

        tr_loader = DataLoader(make_ds(tr_df, True), batch_size=BEST_BENCHMARK["batch_size"], shuffle=True)
        va_loader = DataLoader(make_ds(va_df, False), batch_size=BEST_BENCHMARK["batch_size"], shuffle=False)

        model = MLPBenchmarkModel(
            input_dim=len(INPUT_COLS_BENCH),
            output_dim=len(TARGET_COLS),
            hidden_layers=BEST_BENCHMARK["n_layers"],
            neurons_per_layer=[BEST_BENCHMARK["n_units_l0"]],
            activation_fn=BEST_BENCHMARK["activation"],
            dropout_rate=BEST_BENCHMARK["dropout"],
        ).to(DEVICE)

        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=BEST_BENCHMARK["lr"], momentum=0.9)

        best = float("inf")
        best_target_mses = [float("inf")] * len(TARGET_COLS)
        best_real_target_mses = [float("inf")] * len(TARGET_COLS)
        patience=10; no_improve=0
        for _ in range(500):
            model.train()
            for batch in tr_loader:
                optimizer.zero_grad()
                inputs = batch["input"].to(DEVICE)
                targets = batch["target"].to(DEVICE)
                
                pred = model(inputs)
                loss = criterion(pred, targets)
                loss.backward()
                if isinstance(optimizer, torch.optim.SGD):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            # validation
            model.eval()
            vals = []  # Keep normalized MSE for training decisions
            target_vals = [[] for _ in range(len(TARGET_COLS))]  # Normalized MSE per target
            real_vals = [[] for _ in range(len(TARGET_COLS))]  # Real unit MSE per target
            
            with torch.no_grad():
                for batch in va_loader:
                    inputs = batch["input"].to(DEVICE)
                    targets = batch["target"].to(DEVICE)
                    preds = model(inputs)
                    
                    # Overall normalized MSE
                    vals.append(criterion(preds, targets).item())
                    
                    # Per-target normalized MSE
                    target_mses = target_specific_mse(preds, targets)
                    for i, mse in enumerate(target_mses):
                        target_vals[i].append(mse)
                    
                    # Real-unit MSE per target
                    real_target_mses = calculate_real_mse(preds, targets, y_scaler)
                    for i, mse in enumerate(real_target_mses):
                        real_vals[i].append(mse)
            
            # Calculate mean MSE (normalized and real)
            cur = np.mean(vals)
            norm_target_means = [np.mean(t_vals) for t_vals in target_vals]
            real_target_means = [np.mean(r_vals) for r_vals in real_vals]
            
            # Track best performance (using normalized MSE for decision)
            if cur < best:
                best = cur
                best_target_mses = norm_target_means  # normalized MSE per target
                best_real_target_mses = real_target_means  # real unit MSE per target
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
                    
        logger.info(f"Fold {fold}/{k}  MSE={best:.5f} (normalized)")
        logger.info(f"Real units MSE: " + 
                   " | ".join([f"{TARGET_COLS[i]}={mse:.4f}" for i, mse in enumerate(best_real_target_mses)]))
        
        scores.append(best)
        fold_scores.append(best)
        
        # Add target-specific scores
        for i, mse in enumerate(best_target_mses):
            target_scores[i].append(mse)
        for i, mse in enumerate(best_real_target_mses):
            real_target_scores[i].append(mse)
    
    # Overall results
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    logger.info(f"Benchmark CV →  {mean_score:.5f} ± {std_score:.5f}")
    
    # Target-specific results
    for i, target in enumerate(TARGET_COLS):
        target_mean = np.mean(target_scores[i])
        target_std = np.std(target_scores[i])
        logger.info(f"  {target} →  {target_mean:.5f} ± {target_std:.5f}")
        
    logger.info("")
    
    return mean_score, fold_scores, target_scores, real_target_scores

# --------------------------------------------------------------------------
# 2. SUB-NETWORK Cross validation 
# --------------------------------------------------------------------------
def run_subnet_cv(k: int = 5):
    logger.info("\n=== SUB-NETWORK SYSTEM ===")
    full_df = load_product_dataframe(MAIN_FOLDER_SUBNET)

    scores = []
    fold_scores = []
    # Create lists for each target
    target_scores = [[] for _ in range(len(TARGET_COLS))]
    real_target_scores = [[] for _ in range(len(TARGET_COLS))]
    
    encoder = load_ae_subnet(str(AE_WEIGHTS_SUBNET), DEVICE, latent_dim=256)
    
    for fold, (tr, va) in enumerate(grouped_kfold_indices(full_df, k), 1):
        tr_df, va_df = full_df.iloc[tr].reset_index(drop=True), full_df.iloc[va].reset_index(drop=True)
        
        # Print validation fold products
        val_products = sorted(va_df["product_name"].unique())
        logger.info(f"Fold {fold} validation products: {val_products}")

        # Scale process inputs
        p_scaler = StandardScaler().fit(tr_df[PROC_COLS])
        p_scaler.scale_[p_scaler.scale_ == 0] = 1.0
        
        # Scale targets
        y_scaler = StandardScaler().fit(tr_df[TARGET_COLS])
        y_scaler.scale_[y_scaler.scale_ == 0] = 1.0

        def make_ds(df, is_train):
            return SubnetDataset(
                df,                      # DataFrame directly
                MAIN_FOLDER_SUBNET,      # Root folder path
                encoder,                 # Encoder model
                PROC_COLS,               # Process columns
                TARGET_COLS,             # Target columns
                p_scaler,                # Process scaler
                y_scaler,                # Target scaler
                DEVICE                   # Device
            )

        tr_ds = make_ds(tr_df, True)
        va_ds = make_ds(va_df, False)
        
        tr_loader = DataLoader(tr_ds, batch_size=SUB_ARCH["batch_size"], shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=SUB_ARCH["batch_size"], shuffle=False)

        model = HierarchicalQualityPredictor(
            latent_dim=encoder.fc_mu.out_features,
            proc_dim=len(PROC_COLS),
            lat_hidden=SUB_ARCH["lat_hidden"],
            proc_hidden=SUB_ARCH["proc_hidden"],
            fuse_hidden=SUB_ARCH["fuse_hidden"],
            lat_act=SUB_ARCH["lat_act"],
            proc_act=SUB_ARCH["proc_act"],
            fuse_act=SUB_ARCH["fuse_act"],
            lat_drop=SUB_ARCH["lat_drop"],
            proc_drop=SUB_ARCH["proc_drop"],
            fuse_drop=SUB_ARCH["fuse_drop"],
        ).to(DEVICE)

        criterion = nn.MSELoss()
        if SUB_ARCH["opt"].lower() == "nadam":
            from torch.optim import Adam
            optimizer = Adam(model.parameters(), lr=SUB_ARCH["lr"], betas=(0.9, 0.999), weight_decay=1e-5)
        else:
            from torch.optim import SGD
            optimizer = SGD(model.parameters(), lr=SUB_ARCH["lr"], momentum=0.9)

        best = float("inf")
        best_target_mses = [float("inf")] * len(TARGET_COLS)
        best_real_target_mses = [float("inf")] * len(TARGET_COLS)
        patience=15; no_improve=0
        
        for _ in range(500):
            model.train()
            for batch in tr_loader:
                optimizer.zero_grad()
                lat_vecs = batch["lat"].to(DEVICE)
                proc_params = batch["proc"].to(DEVICE)
                targets = batch["target"].to(DEVICE)
                
                pred = model(lat_vecs, proc_params)
                loss = criterion(pred, targets)
                loss.backward()
                optimizer.step()
                
            # validation
            model.eval()
            vals = []  # Keep normalized MSE for training decisions
            target_vals = [[] for _ in range(len(TARGET_COLS))]  # Normalized MSE per target
            real_vals = [[] for _ in range(len(TARGET_COLS))]  # Real unit MSE per target
            
            with torch.no_grad():
                for batch in va_loader:
                    lat_vecs = batch["lat"].to(DEVICE)
                    proc_params = batch["proc"].to(DEVICE)
                    targets = batch["target"].to(DEVICE)
                    
                    preds = model(lat_vecs, proc_params)
                    
                    # Overall normalized MSE
                    vals.append(criterion(preds, targets).item())
                    
                    # Per-target normalized MSE
                    target_mses = target_specific_mse(preds, targets)
                    for i, mse in enumerate(target_mses):
                        target_vals[i].append(mse)
                    
                    # Real-unit MSE per target
                    real_target_mses = calculate_real_mse(preds, targets, y_scaler)
                    for i, mse in enumerate(real_target_mses):
                        real_vals[i].append(mse)
            
            # Calculate mean MSE (normalized and real)
            cur = np.mean(vals)
            norm_target_means = [np.mean(t_vals) for t_vals in target_vals]
            real_target_means = [np.mean(r_vals) for r_vals in real_vals]
            
            # Track best performance (using normalized MSE for decision)
            if cur < best:
                best = cur
                best_target_mses = norm_target_means 
                best_real_target_mses = real_target_means
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
                    
        logger.info(f"Fold {fold}/{k}  MSE={best:.5f} (normalized)")
        logger.info(f"Real units MSE: " + 
                   " | ".join([f"{TARGET_COLS[i]}={mse:.4f}" for i, mse in enumerate(best_real_target_mses)]))
        
        scores.append(best)
        fold_scores.append(best)
        
        # Add target-specific scores
        for i, mse in enumerate(best_target_mses):
            target_scores[i].append(mse)
        for i, mse in enumerate(best_real_target_mses):
            real_target_scores[i].append(mse)
    
    # Overall results
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    logger.info(f"SubNetwork CV →  {mean_score:.5f} ± {std_score:.5f}")
    
    # Target-specific results
    for i, target in enumerate(TARGET_COLS):
        target_mean = np.mean(target_scores[i])
        target_std = np.std(target_scores[i])
        logger.info(f"  {target} →  {target_mean:.5f} ± {target_std:.5f}")
        
    logger.info("")
    
    return mean_score, fold_scores, target_scores, real_target_scores

# --------------------------------------------------------------------------
# 3. THESIS SYSTEM Cross validation
# --------------------------------------------------------------------------
def run_thesis_cv(k: int = 5):
    logger.info("\n=== THESIS SYSTEM ===")
    full_df = load_product_dataframe(MAIN_FOLDER_THESIS)

    scores = []
    fold_scores = []
    # Create lists for each target
    target_scores = [[] for _ in range(len(TARGET_COLS))]
    real_target_scores = [[] for _ in range(len(TARGET_COLS))]
    
    encoder = load_ae_subnet(str(AE_WEIGHTS_SUBNET), DEVICE, latent_dim=256)
    
    for fold, (tr, va) in enumerate(grouped_kfold_indices(full_df, k), 1):
        tr_df, va_df = full_df.iloc[tr].reset_index(drop=True), full_df.iloc[va].reset_index(drop=True)
        
        # Print validation fold products
        val_products = sorted(va_df["product_name"].unique())
        logger.info(f"Fold {fold} validation products: {val_products}")

        # Scale process inputs 
        p_scaler = StandardScaler().fit(tr_df[PROC_COLS])
        p_scaler.scale_[p_scaler.scale_ == 0] = 1.0
        
        # Scale targets
        y_scaler = StandardScaler().fit(tr_df[TARGET_COLS])
        y_scaler.scale_[y_scaler.scale_ == 0] = 1.0

        def make_ds(df, is_train):
            return ThesisDataset(
                df,                      # DataFrame directly
                MAIN_FOLDER_THESIS,      # Root folder path
                encoder,                 # Encoder model
                PROC_COLS,               # Process columns
                TARGET_COLS,             # Target columns
                p_scaler,                # Process scaler
                y_scaler,                # Target scaler
                DEVICE                   # Device
            )

        tr_ds = make_ds(tr_df, True)
        va_ds = make_ds(va_df, False)
        
        tr_loader = DataLoader(tr_ds, batch_size=THESIS_ARCH["batch_size"], shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=THESIS_ARCH["batch_size"], shuffle=False)

        model = MLPQualityPredictor(
            input_dim=encoder.fc_mu.out_features + len(PROC_COLS),
            output_dim=len(TARGET_COLS),
            hidden_layers=THESIS_ARCH["hidden_layers"],
            neurons_per_layer=THESIS_ARCH["neurons"],
            activation=THESIS_ARCH["act"],
            dropout=THESIS_ARCH["dropout"],
        ).to(DEVICE)

        criterion = make_loss_thesis(THESIS_ARCH["loss_fn"])
        optimizer = make_opt_thesis(
            THESIS_ARCH["opt"],
            model.parameters(),
            lr=THESIS_ARCH["lr"],
        )

        best = float("inf")
        best_target_mses = [float("inf")] * len(TARGET_COLS)
        best_real_target_mses = [float("inf")] * len(TARGET_COLS)
        patience=15; no_improve=0
        
        for _ in range(500):
            model.train()
            for batch in tr_loader:
                optimizer.zero_grad()
                inputs = batch["input"].to(DEVICE)
                targets = batch["target"].to(DEVICE)
                
                pred = model(inputs)
                loss = criterion(pred, targets)
                loss.backward()
                optimizer.step()
                
            # validation
            model.eval()
            vals = []  # Keep normalized MSE for training decisions
            target_vals = [[] for _ in range(len(TARGET_COLS))]  # Normalized MSE per target
            real_vals = [[] for _ in range(len(TARGET_COLS))]  # Real unit MSE per target
            
            with torch.no_grad():
                for batch in va_loader:
                    inputs = batch["input"].to(DEVICE)
                    targets = batch["target"].to(DEVICE)
                    
                    preds = model(inputs)
                    
                    # Overall normalized MSE
                    vals.append(nn.MSELoss()(preds, targets).item())  # Always use MSE for evaluation
                    
                    # Per-target normalized MSE
                    target_mses = target_specific_mse(preds, targets)
                    for i, mse in enumerate(target_mses):
                        target_vals[i].append(mse)
                    
                    # Real-unit MSE per target
                    real_target_mses = calculate_real_mse(preds, targets, y_scaler)
                    for i, mse in enumerate(real_target_mses):
                        real_vals[i].append(mse)
            
            # Calculate mean MSE (normalized and real)
            cur = np.mean(vals)
            norm_target_means = [np.mean(t_vals) for t_vals in target_vals]
            real_target_means = [np.mean(r_vals) for r_vals in real_vals]
            
            # Track best performance (using normalized MSE for decision)
            if cur < best:
                best = cur
                best_target_mses = norm_target_means 
                best_real_target_mses = real_target_means
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
                    
        logger.info(f"Fold {fold}/{k}  MSE={best:.5f} (normalized)")
        logger.info(f"Real units MSE: " + 
                   " | ".join([f"{TARGET_COLS[i]}={mse:.4f}" for i, mse in enumerate(best_real_target_mses)]))
        
        scores.append(best)
        fold_scores.append(best)
        
        # Add target-specific scores
        for i, mse in enumerate(best_target_mses):
            target_scores[i].append(mse)
        for i, mse in enumerate(best_real_target_mses):
            real_target_scores[i].append(mse)
    
    # Overall results
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    logger.info(f"Thesis CV →  {mean_score:.5f} ± {std_score:.5f}")
    
    # Target-specific results
    for i, target in enumerate(TARGET_COLS):
        target_mean = np.mean(target_scores[i])
        target_std = np.std(target_scores[i])
        logger.info(f"  {target} →  {target_mean:.5f} ± {target_std:.5f}")
        
    logger.info("")
    
    return mean_score, fold_scores, target_scores, real_target_scores

# --------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Create a timestamped results file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"kfold_results_{timestamp}.txt"
    
    # Initialize the file with header
    write_results_to_file(results_file, "=" * 80)
    write_results_to_file(results_file, f"K-FOLD CROSS-VALIDATION RESULTS ({N_REPEATS} REPEATS)")
    write_results_to_file(results_file, f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    write_results_to_file(results_file, "=" * 80 + "\n")
    
    all_results = {"benchmark": [], "subnet": [], "thesis": []}
    fold_results = {"benchmark": [], "subnet": [], "thesis": []}
    
    # Create target-specific result containers
    target_results = {
        "benchmark": {t: [] for t in TARGET_COLS},
        "subnet": {t: [] for t in TARGET_COLS},
        "thesis": {t: [] for t in TARGET_COLS}
    }
    
    # Additional storage for real-unit metrics
    real_target_results = {
        "benchmark": {t: [] for t in TARGET_COLS},
        "subnet": {t: [] for t in TARGET_COLS},
        "thesis": {t: [] for t in TARGET_COLS}
    }
    
    for repeat in range(N_REPEATS):
        repeat_header = f"\n==========  REPEAT {repeat+1}/{N_REPEATS}  =========="
        logger.info(repeat_header)  
        write_results_to_file(results_file, f"\n{'='*30} REPEAT {repeat+1}/{N_REPEATS} {'='*30}")
        
        # Set new seed for THIS REPEAT's training randomness
        set_training_seeds(repeat + 1000)
        
        # Run each system and log results to both console and file
        bench_score, bench_folds, bench_targets, bench_real_targets = run_benchmark_cv()
        all_results["benchmark"].append(bench_score)
        fold_results["benchmark"].extend([(repeat+1, i+1, score) for i, score in enumerate(bench_folds)])
        write_results_to_file(results_file, f"Benchmark System: {bench_score:.5f}")
        # Add target results
        for i, target in enumerate(TARGET_COLS):
            target_results["benchmark"][target].append(np.mean(bench_targets[i]))
            real_target_results["benchmark"][target].append(np.mean(bench_real_targets[i]))
            write_results_to_file(results_file, f"  {target}: {np.mean(bench_targets[i]):.5f}")
            write_results_to_file(results_file, f"  {target} (real units): {np.mean(bench_real_targets[i]):.4f}")
        
        subnet_score, subnet_folds, subnet_targets, subnet_real_targets = run_subnet_cv()
        all_results["subnet"].append(subnet_score)
        fold_results["subnet"].extend([(repeat+1, i+1, score) for i, score in enumerate(subnet_folds)])
        write_results_to_file(results_file, f"Sub-network System: {subnet_score:.5f}")
        # Add target results
        for i, target in enumerate(TARGET_COLS):
            target_results["subnet"][target].append(np.mean(subnet_targets[i]))
            real_target_results["subnet"][target].append(np.mean(subnet_real_targets[i]))
            write_results_to_file(results_file, f"  {target}: {np.mean(subnet_targets[i]):.5f}")
            write_results_to_file(results_file, f"  {target} (real units): {np.mean(subnet_real_targets[i]):.4f}")
        
        thesis_score, thesis_folds, thesis_targets, thesis_real_targets = run_thesis_cv()
        all_results["thesis"].append(thesis_score)
        fold_results["thesis"].extend([(repeat+1, i+1, score) for i, score in enumerate(thesis_folds)])
        write_results_to_file(results_file, f"Thesis System: {thesis_score:.5f}")
        # Add target results
        for i, target in enumerate(TARGET_COLS):
            target_results["thesis"][target].append(np.mean(thesis_targets[i]))
            real_target_results["thesis"][target].append(np.mean(thesis_real_targets[i]))
            write_results_to_file(results_file, f"  {target}: {np.mean(thesis_targets[i]):.5f}")
            write_results_to_file(results_file, f"  {target} (real units): {np.mean(thesis_real_targets[i]):.4f}")
    
    # Write all fold results
    write_results_to_file(results_file, "\n\n" + "="*80)
    write_results_to_file(results_file, "DETAILED FOLD RESULTS")
    write_results_to_file(results_file, "="*80)
    
    for system, results in fold_results.items():
        write_results_to_file(results_file, f"\n{system.capitalize()} System:")
        write_results_to_file(results_file, "-"*40)
        write_results_to_file(results_file, "Repeat | Fold | MSE Score")
        write_results_to_file(results_file, "-"*40)
        for repeat, fold, score in results:
            write_results_to_file(results_file, f"{repeat:6d} | {fold:4d} | {score:.5f}")
    
    # Calculate and display final statistics across repeats
    final_header = "\n\n========== FINAL RESULTS (ACROSS REPEATS) =========="
    logger.info(final_header)
    write_results_to_file(results_file, "\n\n" + "="*80)
    write_results_to_file(results_file, "FINAL RESULTS (ACROSS REPEATS)")
    write_results_to_file(results_file, "="*80)
    
    # Overall results
    for name, runs in all_results.items():
        runs = np.array(runs)
        mean, std = runs.mean(), runs.std()
        
        console_msg = f"{name.capitalize():10} →  {mean:.5f} ± {std:.5f} (across {N_REPEATS} repeats)"
        logger.info(console_msg)
        
        file_msg = f"{name.capitalize():10} ->  {mean:.5f} +/- {std:.5f} (across {N_REPEATS} repeats)"
        write_results_to_file(results_file, file_msg)
        
        min_max_msg = f"{' '*12}Min: {runs.min():.5f}, Max: {runs.max():.5f}"
        logger.info(min_max_msg)
        write_results_to_file(results_file, min_max_msg)
    
    # Target-specific final results
    write_results_to_file(results_file, "\n\n" + "="*80)
    write_results_to_file(results_file, "TARGET-SPECIFIC RESULTS (NORMALIZED, ACROSS REPEATS)")
    write_results_to_file(results_file, "="*80)
    
    logger.info("\n========== TARGET-SPECIFIC RESULTS (NORMALIZED) ==========")
    
    for target in TARGET_COLS:
        logger.info(f"\nTarget: {target}")
        write_results_to_file(results_file, f"\nTarget: {target}")
        write_results_to_file(results_file, "-"*40)
        
        for system in ["benchmark", "subnet", "thesis"]:
            runs = np.array(target_results[system][target])
            mean, std = runs.mean(), runs.std()
            
            console_msg = f"{system.capitalize():10} →  {mean:.5f} ± {std:.5f}"
            logger.info(console_msg)
            
            file_msg = f"{system.capitalize():10} ->  {mean:.5f} +/- {std:.5f}"
            write_results_to_file(results_file, file_msg)
    
    # Target-specific real units final results
    write_results_to_file(results_file, "\n\n" + "="*80)
    write_results_to_file(results_file, "TARGET-SPECIFIC RESULTS IN REAL UNITS (ACROSS REPEATS)")
    write_results_to_file(results_file, "="*80)

    logger.info("\n========== TARGET-SPECIFIC RESULTS (REAL UNITS) ==========")

    for target in TARGET_COLS:
        logger.info(f"\nTarget: {target}")
        write_results_to_file(results_file, f"\nTarget: {target}")
        write_results_to_file(results_file, "-"*40)
        
        for system in ["benchmark", "subnet", "thesis"]:
            runs = np.array(real_target_results[system][target])
            mean, std = runs.mean(), runs.std()
            
            console_msg = f"{system.capitalize():10} →  {mean:.4f} ± {std:.4f} (real units)"
            logger.info(console_msg)
            
            file_msg = f"{system.capitalize():10} ->  {mean:.4f} +/- {std:.4f} (real units)"
            write_results_to_file(results_file, file_msg)
    
    write_results_to_file(results_file, f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Detailed results saved to: {results_file}")

