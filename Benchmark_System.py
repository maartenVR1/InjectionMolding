import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna
from optuna.trial import Trial
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import logging

"""
This code only works when the training data is stored in a specific format:

TrainingData_Benchmark_System/
├── product1/
│   ├── product1.xlsx
├── product2/
│   ├── product2.xlsx
└── product3/
    ├── product3.xlsx

"""

# logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

class BenchmarkDataset(Dataset):
    """
    Dataset for benchmark injection molding quality prediction.
    Directly uses geometric features and process parameters as input, no autoencoder invovled.
    """
    def __init__(self, data_csv, input_columns, target_columns, transform=None, target_transform=None, fit_transform=False):
        """
        Args:
            data_csv (str): Path to CSV containing geometric features, process parameters, and quality metrics
            input_columns (list): Column names for input features (geometric + process parameters)
            target_columns (list): Column names for target quality metrics
            transform (callable, optional): Optional transform to be applied on input features
            target_transform (callable, optional): Optional transform to be applied on target values
        """
        self.data = pd.read_csv(data_csv)
        self.input_columns = input_columns
        self.target_columns = target_columns
        self.transform = transform
        self.target_transform = target_transform
        self.fit_transform = fit_transform
        
        # Normalize input features
        if self.transform:
            if fit_transform:
                self.inputs = self.transform.fit_transform(self.data[self.input_columns])
            else:
                self.inputs = self.transform.transform(self.data[self.input_columns])
        else:
            self.inputs = self.data[self.input_columns].values
            
        # Normalize target values because they are in different orders of magnitude
        if self.target_transform:
            if fit_transform:
                self.targets = self.target_transform.fit_transform(self.data[self.target_columns])
            else:
                self.targets = self.target_transform.transform(self.data[self.target_columns])
        else:
            self.targets = self.data[self.target_columns].values
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx): # just a standard dataset method to get an item by index
        input_features = self.inputs[idx]
        target = self.targets[idx]
        
        return {
            'input': torch.tensor(input_features, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32)
        }

class MLPBenchmarkModel(nn.Module): 
    """
    MLP architecture for benchmark injection molding quality prediction.
    """
    def __init__(self, input_dim, output_dim, hidden_layers, neurons_per_layer, 
                 activation_fn='relu', dropout_rate=0.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Set up activation function, selected the most frequently used ones, u can extend this if you want to test more
        if activation_fn == 'relu':
            activation = nn.ReLU()
        elif activation_fn == 'leaky_relu':
            activation = nn.LeakyReLU()
        elif activation_fn == 'elu':
            activation = nn.ELU()
        elif activation_fn == 'tanh':
            activation = nn.Tanh()
        elif activation_fn == 'selu':
            activation = nn.SELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")
        
        # Build the network layer by layer in a list
        layers = []
        
        # Input layer, this is a way to dynamically set the number of layers and its neurons per layer
        input_size = input_dim
        for i in range(hidden_layers):
            layers.append(nn.Linear(input_size, neurons_per_layer[i]))
            layers.append(activation)
            if dropout_rate > 0: #I'm not going to set different dropout rates for different layers, 
                #in fear of overfitting on the validation set
                layers.append(nn.Dropout(dropout_rate))
            input_size = neurons_per_layer[i]
        
        # Output layer
        layers.append(nn.Linear(input_size, output_dim))
        
        self.model = nn.Sequential(*layers) #* unpacks the list of layers into the constructor of nn.Sequential, u dont wnant a list
    
    def forward(self, x): 
        return self.model(x) # the model method comes from nn.Sequential, which is a container for the layers
        # it takes care of the forward pass through the network

class BenchmarkTrainer:
    """
    Trainer class for the benchmark MLP model with Optuna hyperparameter optimization. This is the main class with a tonne of helper functions for clarity.
    """
    def __init__(self, 
                 main_folder,
                 input_columns=['Cavity Volume', 'Cavity Surface Area', 'Projection Area ZX', 
                               'Projection Area YZ', 'Gate Location to Furthest Point in X direction',
                               'Gate Location to Furthest Point in Y direction', 
                               'Gate Location to Furthest Point in Z direction', 'Gate Distance to COM',
                               'Mold Surface Temperature', 'Melt Temperature', 
                               'Packing Pressure', 'Injection Time'],
                 target_columns=['Cavity Weight', 'Maximum Warpage', 'Volumetric Shrinkage'],
                 batch_size_range=(8, 128),
                 random_state=42,
                 n_trials=100,
                 study_name='benchmark_quality_prediction',
                 study_storage=None,
                 output_dir="benchmark_results"
                 ):
        
        self.main_folder = main_folder
        self.input_columns = input_columns
        self.target_columns = target_columns
        self.batch_size_range = batch_size_range
        self.random_state = random_state
        self.n_trials = n_trials
        self.study_name = study_name
        self.study_storage = study_storage
        self.output_dir = output_dir
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output files will be saved to: {os.path.abspath(self.output_dir)}")
        
        # Load data - but don't create loaders yet
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare datasets using the canonical product split from JSON"""
        # Load the canonical product split
        import json
        with open("canonical_product_split.json", "r") as f:
            product_split = json.load(f)
        
        # Log which products are in each split
        logger.info("Using canonical product split:")
        logger.info(f"Train products: {len(product_split['train_products'])} - {product_split['train_products']}")
        logger.info(f"Val products: {len(product_split['val_products'])} - {product_split['val_products']}")
        logger.info(f"Test products: {len(product_split['test_products'])} - {product_split['test_products']}")
        
        # Load all data from multiple subfolders of all producsts
        data = self._load_multi_folder_data()
        
        # Filter data based on product split
        train_data = data[data["product_name"].isin(product_split["train_products"])].reset_index(drop=True)
        val_data = data[data["product_name"].isin(product_split["val_products"])].reset_index(drop=True)
        test_data = data[data["product_name"].isin(product_split["test_products"])].reset_index(drop=True)
        
        # Save split datasets, just for dubble checks
        train_csv_path = os.path.join(self.output_dir, 'benchmark_train_data.csv') 
        train_data.to_csv(train_csv_path, index=False)
        val_csv_path = os.path.join(self.output_dir, 'benchmark_val_data.csv')
        val_data.to_csv(val_csv_path, index=False)
        test_csv_path = os.path.join(self.output_dir, 'benchmark_test_data.csv')
        test_data.to_csv(test_csv_path, index=False)

        # training data scaler, normalization of input features and target values
        self.input_scaler = StandardScaler()
        self.input_scaler.fit(train_data[self.input_columns])
        
        # Division by zer handling (gate location is constant)
        self.input_scaler.scale_[self.input_scaler.scale_ == 0] = 1.0

        # scaler for target values, u could also not normlalize these targets but i think taht would be a bad idea
        self.target_scaler = StandardScaler()
        self.target_scaler.fit(train_data[self.target_columns])
        
        # Handle potential constant columns
        if hasattr(self.target_scaler, 'scale_'):
            self.target_scaler.scale_[self.target_scaler.scale_ == 0] = 1.0
        
        # Create datasets with pre-fitted scalers
        self.train_dataset = BenchmarkDataset(
            train_csv_path, 
            self.input_columns, self.target_columns, 
            transform=self.input_scaler,
            target_transform=self.target_scaler,  
            fit_transform=False
        )
        
        self.val_dataset = BenchmarkDataset(
            val_csv_path, 
            self.input_columns, self.target_columns, 
            transform=self.input_scaler,
            target_transform=self.target_scaler,  
            fit_transform=False
        )
        
        self.test_dataset = BenchmarkDataset(
            test_csv_path, 
            self.input_columns, self.target_columns, 
            transform=self.input_scaler,
            target_transform=self.target_scaler,  
            fit_transform=False
        )
        
        # Calculate input and output dimensions
        self.input_dim = len(self.input_columns)
        self.output_dim = len(self.target_columns)
        
        logger.info(f"Data prepared. Train: {len(self.train_dataset)}, "
                   f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
        logger.info(f"Input dim: {self.input_dim}, Output dim: {self.output_dim}")
    
    def _create_data_loaders(self, batch_size):
        """Create data loaders with the specified batch size"""
        train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )
        
        val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False
        )
        
        test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False
        )
        
        return train_loader, val_loader, test_loader
    
    def _create_model(self, trial: Trial):
        """Create MLP model with parameters suggested by Optuna"""
        # Suggest number of hidden layers 
        n_layers = trial.suggest_int('n_layers', 1, 10)
        
        # Suggest number of neurons for each layer 
        neurons = []
        for i in range(n_layers):
            neurons.append(trial.suggest_int(f'n_units_l{i}', 16, 512, log=True)) # here we create a list of the amount of neurons per layer
        
        # Suggest activation function
        activation = trial.suggest_categorical('activation', 
                                              ['relu', 'leaky_relu', 'elu', 'tanh', 'selu'])
        
        # Suggest dropout rate, could increase this range
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        
        # Create model
        model = MLPBenchmarkModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_layers=n_layers,
            neurons_per_layer=neurons,
            activation_fn=activation,
            dropout_rate=dropout
        ).to(self.device)
        
        return model
    
    def _train_model(self, trial, model):
        """Train the model and return best validation loss"""
        
        # Suggest optimizer (extended selection) - MOVED UP
        optimizer_name = trial.suggest_categorical('optimizer', 
                             ['AdamW', 'Adam', 'RMSprop', 'SGD', 
                              'NAdam', 'RAdam', 'Adamax', 'Adagrad'])
        
        # Suggest learning rate with adjusted range 
        if optimizer_name == 'SGD':  
            # SGD benefits from higher learning rates
            lr = trial.suggest_float('lr', 1e-2, 1e-1, log=True)  
        else:
            # Adaptive optimizers (Adam, AdamW, etc)
            lr = trial.suggest_float('lr', 1e-4, 3e-2, log=True)
        
        # Create optimizer based on suggestion
        if optimizer_name == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=lr)
        elif optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=lr)
        elif optimizer_name == 'NAdam':
            optimizer = optim.NAdam(model.parameters(), lr=lr)
        elif optimizer_name == 'RAdam':
            optimizer = optim.RAdam(model.parameters(), lr=lr)
        elif optimizer_name == 'Adamax':
            optimizer = optim.Adamax(model.parameters(), lr=lr)
        elif optimizer_name == 'Adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=lr)
        else:  # SGD
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
        # Training loss can vary between trials
        loss_name = trial.suggest_categorical('loss', ['MSE', 'L1', 'SmoothL1', 'Huber', 'LogCosh'])
        
        if loss_name == 'MSE':
            criterion = nn.MSELoss()
        elif loss_name == 'L1':
            criterion = nn.L1Loss()
        elif loss_name == 'SmoothL1':
            criterion = nn.SmoothL1Loss()
        elif loss_name == 'Huber':
            criterion = nn.HuberLoss(delta=1.0)
        else:  # LogCosh
            def log_cosh_loss(input, target):
                return torch.mean(torch.log(torch.cosh(input - target)))
            criterion = log_cosh_loss
            
        # Add a consistent validation metric, different from the training loss
        fixed_validation_metric = nn.MSELoss()  # Always use MSE for validation, like other systems as well

        num_epochs = 250
        patience = 25 
        
        # Early stopping parameters
        best_val_loss = float('inf')
        counter = 0
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            training_loss_sum = 0.0 
            
            for batch in self.train_loader:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Forward pass
                outputs = model(inputs)
                trial_specific_loss = criterion(outputs, targets)  
                
                # Backward and optimize
                optimizer.zero_grad()
                trial_specific_loss.backward() #this propagates the loss backwards through the network
                # update the weights 
                optimizer.step()    
                
                training_loss_sum += trial_specific_loss.item()

            # Validation phase, this code does validate 
            # the network on the validation set and uses that validation performance to guide hyperparameter selection through Optuna
            model.eval()
            validation_loss_sum = 0.0  
            
            with torch.no_grad():
                for batch in self.val_loader:
                    inputs = batch['input'].to(self.device)
                    targets = batch['target'].to(self.device)
                    
                    outputs = model(inputs)
                    # Use the fixed metric for validation
                    validation_loss = fixed_validation_metric(outputs, targets)  
                    
                    validation_loss_sum += validation_loss.item()

            # Calculate average losses 
            avg_training_loss = training_loss_sum / len(self.train_loader)
            avg_validation_loss = validation_loss_sum / len(self.val_loader)

            # Log progress
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{num_epochs}, "
                           f"Train Loss ({loss_name}): {avg_training_loss:.4f}, "
                           f"Val Loss (MSE): {avg_validation_loss:.4f}")
            
            # Check if current model is best
            if avg_validation_loss < best_val_loss:
                best_val_loss = avg_validation_loss
                counter = 0
                
                # Define both model paths
                best_model_path = os.path.join(self.output_dir, f'benchmark_trial_{trial.number}_best_model.pt')
                global_best_path = os.path.join(self.output_dir, 'benchmark_best_model.pt')
                
                # Save model info for comparison
                model_info = {
                    'trial_number': trial.number,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'loss_name': loss_name,
                    'optimizer_name': optimizer_name,
                }
                
                # Save current trial's best model
                torch.save(model_info, best_model_path)

                global_best_path = os.path.join(self.output_dir, 'benchmark_best_model.pt')

                # Check if this is better than global best
                if os.path.exists(global_best_path):
                    previous_best = torch.load(global_best_path, weights_only=True)
                    if best_val_loss < previous_best['val_loss']: 
                        torch.save(model_info, global_best_path)
                        logger.info(f"New best model found! Trial {trial.number}, Epoch {epoch}, Val Loss: {best_val_loss:.4f}")
                else:
                    torch.save(model_info, global_best_path)
                    logger.info(f"First best model saved. Trial {trial.number}, Epoch {epoch}, Val Loss: {best_val_loss:.4f}")
            else:
                counter += 1
            
            # Early stopping
            if counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Report intermediate metric to Optuna
            trial.report(avg_validation_loss, epoch)
            
            # Handle pruning based on the intermediate value
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned.")
                raise optuna.exceptions.TrialPruned()
        
        return best_val_loss
    
    def _load_multi_folder_data(self):
        """Load data from multiple subfolders of the products, which is how I have it organized"""
        all_data = []
        
        # Scan all product folders
        for product_folder in os.listdir(self.main_folder):
            product_path = os.path.join(self.main_folder, product_folder)
            if not os.path.isdir(product_path):
                continue
                
            # Find the Excel file in this folder
            excel_files = [f for f in os.listdir(product_path) if f.endswith('.xlsx') or f.endswith('.xls')]
            if not excel_files:
                logger.warning(f"No Excel files found in {product_path}")
                continue
                
            # Read the Excel file
            excel_path = os.path.join(product_path, excel_files[0])
            df = pd.read_excel(excel_path)
            
            # Add product name column for tracking
            df['product_name'] = product_folder
                
            all_data.append(df)
        
        # Combine all data
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            return combined_data
        else:
            raise ValueError("No valid data found in subfolders")
    
    def optimize(self):
        """Run Optuna optimization to find best hyperparameters"""
        # Create Optuna study
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.study_storage,
            direction='minimize',
            load_if_exists=True
        )
        
        # Optimize the objective function
        # Optuna creates n trials, and uses the objective method to evaluate each trial
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # Get best trial
        best_trial = study.best_trial
        logger.info(f"Best trial: {best_trial.number}")
        logger.info(f"Value: {best_trial.value}")
        logger.info(f"Params: {best_trial.params}")
        
        # Load best model
        best_model = self._create_model(best_trial)
        best_model_path = os.path.join(self.output_dir, 'benchmark_best_model.pt')
        
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            best_model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from {best_model_path}")
        else:
            logger.warning("No best model file found. This should not happen.")
        
        # Get best trial's batch size
        best_batch_size = best_trial.params['batch_size']
        
        # Create loaders with the best batch size
        _, _, best_test_loader = self._create_data_loaders(best_batch_size)
        
        # Use the correct loader for evaluation
        test_loss, test_metrics = self.evaluate(best_model, best_test_loader)
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Metrics: {test_metrics}")
        
        # Save study results
        self._save_study_results(study)
        
        return best_model, study
    
    def objective(self, trial: Trial):
        
        # Suggest batch size
        batch_size = trial.suggest_int('batch_size', 
                                      self.batch_size_range[0], 
                                      self.batch_size_range[1], 
                                      log=True)
        
        logger.info(f"Trial {trial.number} using batch size: {batch_size}")
        
        # Create data loaders with the suggested batch size
        self.train_loader, self.val_loader, self.test_loader = self._create_data_loaders(batch_size)
        
        # Create model
        model = self._create_model(trial)
        
        # Train model
        best_val_loss = self._train_model(trial, model)
        
        return best_val_loss
    
    def evaluate(self, model, test_loader=None):
        """Evaluate model on test set"""
        model.eval()
        test_loss = 0.0
        
        # Use provided test_loader or fall back to self.test_loader
        if test_loader is None:
            test_loader = self.test_loader
        
        # For detailed metrics calculation
        all_targets_norm = []
        all_outputs_norm = []
        
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                
                # Store normalized targets and outputs
                all_targets_norm.append(targets.cpu().numpy())
                all_outputs_norm.append(outputs.cpu().numpy())
        
        avg_test_loss = test_loss / len(test_loader)
        
        # Convert lists to numpy arrays (still normalized)
        all_targets_norm = np.vstack(all_targets_norm)
        all_outputs_norm = np.vstack(all_outputs_norm)
        
        # Convert back to original scale for metrics
        all_targets = self.inverse_transform_targets(all_targets_norm)
        all_outputs = self.inverse_transform_targets(all_outputs_norm)
        
        # Calculate metrics for each output dimension on original scale
        metrics = {}
        for i, col in enumerate(self.target_columns):
            # Calculate mean absolute error
            mae = np.mean(np.abs(all_targets[:, i] - all_outputs[:, i]))
            # Calculate root mean squared error
            rmse = np.sqrt(np.mean((all_targets[:, i] - all_outputs[:, i])**2))
            # Calculate R² score
            ss_tot = np.sum((all_targets[:, i] - np.mean(all_targets[:, i]))**2)
            ss_res = np.sum((all_targets[:, i] - all_outputs[:, i])**2)
            r2 = 1 - (ss_res / ss_tot)
            
            metrics[col] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        
        return avg_test_loss, metrics
    
    def _save_study_results(self, study):
        """Save study results to files"""
        # Existing CSV saving
        study_stats = {
            'best_value': study.best_value,
            'best_params': study.best_params,
            'best_trial': study.best_trial.number,
            'n_trials': len(study.trials),
            'datetime': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        stats_path = os.path.join(self.output_dir, 'benchmark_study_stats.csv')
        pd.DataFrame([study_stats]).to_csv(stats_path, index=False)
        
        trials_data = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                data = {'trial': trial.number, 'value': trial.value}
                data.update(trial.params)
                trials_data.append(data)
        
        trials_path = os.path.join(self.output_dir, 'benchmark_study_trials.csv')
        pd.DataFrame(trials_data).to_csv(trials_path, index=False)
        
        # txt file with summary of trials
        summary_path = os.path.join(self.output_dir, 'benchmark_study_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Benchmark Study Summary\n")
            f.write("=======================\n\n")
            for trial in study.trials:
                f.write(f"Trial Number: {trial.number}\n")
                f.write(f"  State: {trial.state}\n")
                f.write(f"  Value (Val Loss): {trial.value}\n")
                f.write(f"  Hyperparameters:\n")
                for k, v in trial.params.items():
                    f.write(f"    {k}: {v}\n")
                f.write("\n")  # Blank line between trials
        
        try:
            fig = optuna.visualization.plot_optimization_history(study)
            hist_path = os.path.join(self.output_dir, 'benchmark_optimization_history.png')
            fig.write_image(hist_path)

            fig = optuna.visualization.plot_param_importances(study)
            param_imp_path = os.path.join(self.output_dir, 'benchmark_param_importances.png')
            fig.write_image(param_imp_path)

            fig = optuna.visualization.plot_contour(study, params=list(study.best_params.keys())[:2])
            contour_path = os.path.join(self.output_dir, 'benchmark_contour_plot.png')
            fig.write_image(contour_path)

            logger.info("Study results saved to CSV, TXT, and visualization files.")
        except Exception as e:
            logger.warning(f"Could not create or save visualizations: {e}")
            logger.info("Install kaleido package for visualization support.")
            logger.info("Study results saved to CSV and TXT files only.")

    def inverse_transform_targets(self, normalized_targets):
        """Convert standardized target values back to original scale"""
        if isinstance(normalized_targets, torch.Tensor):
            # Handle torch tensors
            return torch.from_numpy(
                self.target_scaler.inverse_transform(normalized_targets.detach().cpu().numpy())
            ).to(normalized_targets.device)
        else:
            # Handle numpy arrays
            return self.target_scaler.inverse_transform(normalized_targets)

if __name__ == "__main__":
    # Main folder containing product subfolders with Excel files
    MAIN_FOLDER = r"C:\Users\maart\OneDrive - KU Leuven\KUL\MOAI\Master thesis\code\SYSTEM\TrainingData_Benchmark_System"
    
    # Create benchmark trainer
    trainer = BenchmarkTrainer(
        main_folder=MAIN_FOLDER,
        
        input_columns=[
            'Cavity Volume (mm3)', 
            'Cavity Surface Area (mm2)', 
            'Projection Area zx (mm2)',
            'Projection Area yz (mm2)', 
            'gate location to furthest point in x',
            'gate location to furthest point in y', 
            'gate location to furthest point in z', 
            'gate location to COM',
            'Mold Surface Temperature', 
            'Melt Temperature',
            'Packing Pressure', 
            'Injection Time'
        ],

        target_columns=[
            'CavityWeight', 
            'MaxWarp', 
            'VolumetricShrinkage'
        ],
        output_dir="benchmark_results",
        n_trials=5000,
        study_name='benchmark_quality_prediction'
    )
    
    # Run optimization using Optuna
    best_model, study = trainer.optimize()