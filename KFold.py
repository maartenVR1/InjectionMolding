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
    # Architecture parameters
    lat_hidden=[105, 32, 48, 81, 510, 326],
    proc_hidden=[197, 40, 228, 51],
    fuse_hidden=[289],
    
    # Activations
    lat_act="tanh",
    proc_act="tanh",
    fuse_act="selu",
    
    # Dropout rates
    lat_drop=0.0193,
    proc_drop=0.4995,
    fuse_drop=0.1515,
    
    # Per-tower training parameters
    batch_size=28,  
    
    # Tower-specific optimizers
    lat_opt="adagrad",
    proc_opt="adamax", 
    fuse_opt="radam",
    
    # Tower-specific learning rates
    lat_lr=0.000341,
    proc_lr=0.000949,
    fuse_lr=0.024015,
    
    # Tower-specific loss functions
    lat_loss="huber",
    proc_loss="smooth_l1",
    fuse_loss="log_cosh",
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


# Add this helper function to calculate MSE in real units for each target
def calculate_real_mse(y_hat, y, scaler):
    """Calculate MSE for predictions in real units (inverse transformed)"""
    y_hat_real = scaler.inverse_transform(y_hat.cpu().numpy())
    y_real = scaler.inverse_transform(y.cpu().numpy())
    
    # Calculate MSE for each target column
    mse_per_target = []
    for i in range(y_real.shape[1]):
        mse_per_target.append(((y_hat_real[:, i] - y_real[:, i]) ** 2).mean())
    
    return mse_per_target


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
    for fold, (tr, va) in enumerate(grouped_kfold_indices(full_df, k), 1):
        tr_df, va_df = full_df.iloc[tr].reset_index(drop=True), full_df.iloc[va].reset_index(drop=True)
        
        # Print validation fold products
        val_products = sorted(va_df["product_name"].unique())
        logger.info(f"Fold {fold} validation products: {val_products}")

        x_scaler = StandardScaler().fit(tr_df[INPUT_COLS_BENCH])
        x_scaler.scale_[x_scaler.scale_ == 0] = 1.0  # Add this protection line
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
                fit_transform=False  # Always False - we've already fitted above
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

        best = float("inf"); patience=10; no_improve=0
        for _ in range(500):
            model.train()
            for batch in tr_loader:
                optimizer.zero_grad()
                pred = model(batch["input"].to(DEVICE))
                loss = criterion(pred, batch["target"].to(DEVICE))
                loss.backward()
                if isinstance(optimizer, torch.optim.SGD):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            # validation
            model.eval()
            vals = []  # Keep normalized MSE for training decisions
            real_vals = [[] for _ in range(len(TARGET_COLS))]  # For real unit MSE per target

            with torch.no_grad():
                for batch in va_loader:
                    inputs = batch["input"].to(DEVICE)
                    targets = batch["target"].to(DEVICE)
                    preds = model(inputs)
                    
                    # Normalized MSE (for training decisions)
                    vals.append(criterion(preds, targets).item())
                    
                    # Real-unit MSE per target
                    real_mse = calculate_real_mse(preds, targets, y_scaler)
                    for i, mse in enumerate(real_mse):
                        real_vals[i].append(mse)

            # Use normalized MSE for early stopping
            cur = np.mean(vals)
            if cur < best: 
                best = cur
                # Store best real MSE values when normalized MSE is best
                best_real_mse = [np.mean(vals_per_target) for vals_per_target in real_vals]
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience: break

        # Report both normalized and real MSE
        logger.info(f"Fold {fold}/{k}  MSE={best:.5f} (normalized)")
        logger.info(f"  Real units: " + " | ".join([f"{TARGET_COLS[i]}={mse:.4f}" for i, mse in enumerate(best_real_mse)]))
        scores.append(best)
        fold_scores.append(best)
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    logger.info(f"Benchmark CV →  {mean_score:.5f} ± {std_score:.5f}\n")
    return mean_score, scores  # Return both mean and individual fold scores


# --------------------------------------------------------------------------
# 2. SUB‑NETWORK CV
# --------------------------------------------------------------------------

def run_subnet_cv(k: int = 5):
    logger.info("=== SUB‑NETWORK SYSTEM ===")
    
    # Load dataframe and print column names for debugging
    full_df = load_product_dataframe(MAIN_FOLDER_SUBNET)
    logger.info(f"Available columns: {full_df.columns.tolist()}")
    
    # Fix column name mapping if needed
    rename_map = {}
    if 'Packing pressure' in full_df.columns:  # lowercase 'p'
        rename_map['Packing pressure'] = 'Packing Pressure'
    
    if rename_map:
        full_df = full_df.rename(columns=rename_map)
    
    print("About to load autoencoder...")
    ae = load_ae_subnet(str(AE_WEIGHTS_SUBNET), DEVICE, latent_dim=256)
    print("Autoencoder loaded")
    
    scores = []
    fold_scores = []
    for fold, (tr, va) in enumerate(grouped_kfold_indices(full_df, k), 1):
        tr_df, va_df = full_df.iloc[tr].reset_index(drop=True), full_df.iloc[va].reset_index(drop=True)
        
        # Print validation fold products
        val_products = sorted(va_df["product_name"].unique())
        logger.info(f"Fold {fold} validation products: {val_products}")

        scaler_p = StandardScaler().fit(tr_df[PROC_COLS])
        scaler_p.scale_[scaler_p.scale_ == 0] = 1.0
        scaler_t = StandardScaler().fit(tr_df[TARGET_COLS])
        scaler_t.scale_[scaler_t.scale_==0]=1.0

        def make_loader(df, fit, bs, shuffle):
            ds = SubnetDataset(df, MAIN_FOLDER_SUBNET, ae, PROC_COLS, TARGET_COLS, scaler_p, scaler_t, DEVICE)
            return DataLoader(ds, batch_size=bs, shuffle=shuffle)

        tr_loader = make_loader(tr_df, True, SUB_ARCH["batch_size"], True)
        va_loader = make_loader(va_df, False, SUB_ARCH["batch_size"], False)

        model = HierarchicalQualityPredictor(
            latent_dim=ae.fc_mu.out_features,
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

        # Create loss functions for each tower
        def make_loss(loss_type):
            if loss_type.lower() == "mse":
                return nn.MSELoss()
            elif loss_type.lower() in ["mae", "l1"]:
                return nn.L1Loss()
            elif loss_type.lower() == "huber":
                return nn.HuberLoss(delta=1.0)
            elif loss_type.lower() == "smooth_l1":
                return nn.SmoothL1Loss()
            elif loss_type.lower() == "log_cosh":
                # Custom implementation of log-cosh loss
                def log_cosh_loss(pred, target):
                    return torch.mean(torch.log(torch.cosh(pred - target)))
                return log_cosh_loss
            else:
                # Default to MSE
                logger.warning(f"Unknown loss type '{loss_type}', defaulting to MSE")
                return nn.MSELoss()
        
        lat_criterion = make_loss(SUB_ARCH["lat_loss"])
        proc_criterion = make_loss(SUB_ARCH["proc_loss"])
        fuse_criterion = make_loss(SUB_ARCH["fuse_loss"])
        
        # Fixed metric for evaluation
        eval_criterion = nn.MSELoss()  # Always use MSE for evaluation

        # Create optimizer creation function
        def make_optimizer(opt_type, params, lr):
            if opt_type.lower() == "adam":
                return torch.optim.Adam(params, lr=lr)
            elif opt_type.lower() == "adamw":
                return torch.optim.AdamW(params, lr=lr)
            elif opt_type.lower() == "sgd":
                return torch.optim.SGD(params, lr=lr, momentum=0.9)
            elif opt_type.lower() == "adagrad":
                return torch.optim.Adagrad(params, lr=lr)
            elif opt_type.lower() == "rmsprop":
                return torch.optim.RMSprop(params, lr=lr)
            elif opt_type.lower() == "nadam":
                return torch.optim.NAdam(params, lr=lr)  # Correct implementation
            elif opt_type.lower() == "radam":
                return torch.optim.RAdam(params, lr=lr)  # Correct implementation
            else:
                logger.warning(f"Unknown optimizer '{opt_type}', defaulting to Adagrad")
                return torch.optim.Adagrad(params, lr=lr)

        # Create separate optimizers for each tower
        lat_optimizer = make_optimizer(
            SUB_ARCH["lat_opt"], 
            model.lat_tower.parameters(), 
            SUB_ARCH["lat_lr"]
        )
        
        proc_optimizer = make_optimizer(
            SUB_ARCH["proc_opt"], 
            model.proc_tower.parameters(), 
            SUB_ARCH["proc_lr"]
        )
        
        fuse_optimizer = make_optimizer(
            SUB_ARCH["fuse_opt"], 
            [*model.fuse_net.parameters(), *model.out.parameters()], 
            SUB_ARCH["fuse_lr"]
        )

        best = float("inf")
        patience = 10
        no_improve = 0
        for _ in range(500):
            model.train()
            for batch in tr_loader:
                lat_vecs = batch["lat"].to(DEVICE)
                proc_params = batch["proc"].to(DEVICE)
                targets = batch["target"].to(DEVICE)
                
                # Forward pass with tower-specific components
                lat_output = model.lat_tower(lat_vecs)
                proc_output = model.proc_tower(proc_params)
                
                # Concatenate outputs for fusion
                fused = torch.cat([lat_output, proc_output], dim=1)
                fused_output = model.fuse_net(fused)
                pred = model.out(fused_output)
                
                # Update latent tower
                lat_optimizer.zero_grad()
                lat_output = model.lat_tower(lat_vecs)
                proc_output = model.proc_tower(proc_params)
                fused = torch.cat([lat_output, proc_output], dim=1)
                fused_output = model.fuse_net(fused)
                pred = model.out(fused_output)
                lat_loss = lat_criterion(pred, targets)
                lat_loss.backward()  # No retain_graph
                torch.nn.utils.clip_grad_norm_(model.lat_tower.parameters(), max_norm=1.0)
                lat_optimizer.step()

                # Update process tower with a fresh forward pass
                proc_optimizer.zero_grad()
                lat_output = model.lat_tower(lat_vecs)
                proc_output = model.proc_tower(proc_params)
                fused = torch.cat([lat_output, proc_output], dim=1)
                fused_output = model.fuse_net(fused)
                pred = model.out(fused_output)
                proc_loss = proc_criterion(pred, targets)
                proc_loss.backward()  # No retain_graph
                torch.nn.utils.clip_grad_norm_(model.proc_tower.parameters(), max_norm=1.0)
                proc_optimizer.step()

                # Update fusion network with a fresh forward pass
                fuse_optimizer.zero_grad()
                lat_output = model.lat_tower(lat_vecs)
                proc_output = model.proc_tower(proc_params)
                fused = torch.cat([lat_output, proc_output], dim=1)
                fused_output = model.fuse_net(fused)
                pred = model.out(fused_output)
                fuse_loss = fuse_criterion(pred, targets)
                fuse_loss.backward()
                torch.nn.utils.clip_grad_norm_([*model.fuse_net.parameters(), *model.out.parameters()], max_norm=1.0)
                fuse_optimizer.step()

            # val
            model.eval()
            vals = []
            real_vals = [[] for _ in range(len(TARGET_COLS))]

            with torch.no_grad():
                for batch in va_loader:
                    lat_vecs = batch["lat"].to(DEVICE)
                    proc_params = batch["proc"].to(DEVICE)
                    targets = batch["target"].to(DEVICE)
                    
                    # Use standard model forward
                    preds = model(lat_vecs, proc_params)
                    
                    # Always use MSE for evaluation
                    vals.append(eval_criterion(preds, targets).item())
                    
                    # Calculate real-unit MSE for reporting
                    real_mse = calculate_real_mse(preds, targets, scaler_t)
                    for i, mse in enumerate(real_mse):
                        real_vals[i].append(mse)

            cur = np.mean(vals)
            if cur < best:
                best = cur
                best_real_mse = [np.mean(vals_per_target) for vals_per_target in real_vals]
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
                    
        logger.info(f"Fold {fold}/{k}  MSE={best:.5f} (normalized)")
        logger.info(f"  Real units: " + " | ".join([f"{TARGET_COLS[i]}={mse:.4f}" for i, mse in enumerate(best_real_mse)]))
        scores.append(best)
        fold_scores.append(best)
        
    logger.info(f"Sub‑network CV → {np.mean(scores):.5f} ± {np.std(scores):.5f}\n")
    return np.mean(scores), fold_scores


# --------------------------------------------------------------------------
# 3. THESIS / NORMAL SYSTEM CV
# --------------------------------------------------------------------------

def run_thesis_cv(k: int = 5):
    logger.info("=== THESIS / NORMAL SYSTEM ===")
    # Use the same shared loading function
    full_df = load_product_dataframe(MAIN_FOLDER_THESIS)
    
    ae = load_ae_thesis(str(AE_WEIGHTS_THESIS), DEVICE, latent_dim=256)

    scores=[]
    fold_scores = []
    for fold,(tr,va) in enumerate(grouped_kfold_indices(full_df,k),1):
        tr_df, va_df = full_df.iloc[tr].reset_index(drop=True), full_df.iloc[va].reset_index(drop=True)
        
        # Print validation fold products
        val_products = sorted(va_df["product_name"].unique())
        logger.info(f"Fold {fold} validation products: {val_products}")

        scaler_p = StandardScaler().fit(tr_df[PROC_COLS])
        scaler_p.scale_[scaler_p.scale_ == 0] = 1.0
        scaler_t = StandardScaler().fit(tr_df[TARGET_COLS])
        scaler_t.scale_[scaler_t.scale_==0]=1.0

        def make_loader(df, bs, shuffle):
            ds = ThesisDataset(df, MAIN_FOLDER_THESIS, ae, PROC_COLS, TARGET_COLS, scaler_p, scaler_t, DEVICE)
            return DataLoader(ds, batch_size=bs, shuffle=shuffle)

        tr_loader = make_loader(tr_df, THESIS_ARCH["batch_size"], True)
        va_loader = make_loader(va_df, THESIS_ARCH["batch_size"], False)

        model = MLPQualityPredictor(
            input_dim=ae.fc_mu.out_features + len(PROC_COLS),
            output_dim=len(TARGET_COLS),
            hidden_layers=THESIS_ARCH["hidden_layers"],
            neurons_per_layer=THESIS_ARCH["neurons"],
            activation=THESIS_ARCH["act"],
            dropout=THESIS_ARCH["dropout"],
        ).to(DEVICE)

        criterion = make_loss_thesis(THESIS_ARCH["loss_fn"])
        optimizer = make_opt_thesis(THESIS_ARCH["opt"], model.parameters(), THESIS_ARCH["lr"])

        best=float("inf"); patience=10; no_improve=0
        for _ in range(500):
            model.train()
            for batch in tr_loader:
                optimizer.zero_grad()
                pred = model(batch["input"].to(DEVICE))
                loss = criterion(pred, batch["target"].to(DEVICE))
                loss.backward()
                
                optimizer.step()
            # val
            model.eval()
            vals = []
            real_vals = [[] for _ in range(len(TARGET_COLS))]

            with torch.no_grad():
                for batch in va_loader:
                    preds = model(batch["input"].to(DEVICE))
                    vals.append(criterion(preds, batch["target"].to(DEVICE)).item())
                    real_mse = calculate_real_mse(preds, batch["target"], scaler_t)
                    for i, mse in enumerate(real_mse):
                        real_vals[i].append(mse)

            cur=np.mean(vals)
            if cur<best: 
                best=cur
                best_real_mse = [np.mean(vals_per_target) for vals_per_target in real_vals]
                no_improve=0
            else:
                no_improve+=1
                if no_improve>=patience: break
        logger.info(f"Fold {fold}/{k}  MSE={best:.5f} (normalized)")
        logger.info(f"  Real units: " + " | ".join([f"{TARGET_COLS[i]}={mse:.4f}" for i, mse in enumerate(best_real_mse)]))
        scores.append(best)
        fold_scores.append(best)
    logger.info(f"Thesis system CV → {np.mean(scores):.5f} ± {np.std(scores):.5f}\n")
    return np.mean(scores), fold_scores


# --------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("Script starting...")
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
    
    for repeat in range(N_REPEATS):
        repeat_header = f"\n==========  REPEAT {repeat+1}/{N_REPEATS}  =========="
        logger.info(repeat_header)  
        write_results_to_file(results_file, f"\n{'='*30} REPEAT {repeat+1}/{N_REPEATS} {'='*30}")
        
        # Set new seed for THIS REPEAT's training randomness
        set_training_seeds(repeat + 1000)
        
        # Run each system and log results to both console and file
        bench_score, bench_folds = run_benchmark_cv()
        all_results["benchmark"].append(bench_score)
        fold_results["benchmark"].extend([(repeat+1, i+1, score) for i, score in enumerate(bench_folds)])
        write_results_to_file(results_file, f"Benchmark System: {bench_score:.5f}")
        
        subnet_score, subnet_folds = run_subnet_cv()
        all_results["subnet"].append(subnet_score)
        fold_results["subnet"].extend([(repeat+1, i+1, score) for i, score in enumerate(subnet_folds)])
        write_results_to_file(results_file, f"Sub-network System: {subnet_score:.5f}")
        
        thesis_score, thesis_folds = run_thesis_cv()
        all_results["thesis"].append(thesis_score)
        fold_results["thesis"].extend([(repeat+1, i+1, score) for i, score in enumerate(thesis_folds)])
        write_results_to_file(results_file, f"Thesis System: {thesis_score:.5f}")
    
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
    logger.info(final_header)  # Log to console
    write_results_to_file(results_file, "\n\n" + "="*80)
    write_results_to_file(results_file, "FINAL RESULTS (ACROSS REPEATS)")
    write_results_to_file(results_file, "="*80)
    
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
    
    write_results_to_file(results_file, f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Detailed results saved to: {results_file}")

