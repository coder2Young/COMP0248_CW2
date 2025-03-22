import os
import json
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import logging
import numpy as np

class TrainingLogger:
    """
    Utility class for logging training progress and results.
    """
    def __init__(self, log_dir, experiment_name, use_tensorboard=True, log_to_file=True, save_best_model=True, tensorboard_dir=None):
        """
        Initialize the logger.
        
        Args:
            log_dir (str): Directory to save logs
            experiment_name (str): Name of the experiment
            use_tensorboard (bool): Whether to use TensorBoard
            log_to_file (bool): Whether to log to a file
            save_best_model (bool): Whether to save the best model based on validation metrics
            tensorboard_dir (str, optional): Directory to save TensorBoard logs. If None, logs will be saved in log_dir.
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.use_tensorboard = use_tensorboard
        self.log_to_file = log_to_file
        self.save_best_model = save_best_model
        self.tensorboard_dir = tensorboard_dir
        
        # Create log directory
        self.experiment_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Create TensorBoard writer
        if use_tensorboard:
            if tensorboard_dir is not None:
                # Use the common tensorboard directory directly without subdirectories
                os.makedirs(tensorboard_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir=tensorboard_dir, filename_suffix=f"_{experiment_name}")
            else:
                # Use the experiment directory
                self.writer = SummaryWriter(log_dir=self.experiment_dir)
        
        # Set up logging
        if log_to_file:
            self.logger = logging.getLogger(experiment_name)
            self.logger.setLevel(logging.INFO)
            
            # Add file handler
            log_file = os.path.join(self.experiment_dir, 'training.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # Add console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Set formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # Add handlers to logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rate': [],
            'epoch': []
        }
        
        # Best model tracking
        self.best_val_metric = float('inf')  # For loss minimization
        self.best_epoch = 0
    
    def log_hyperparameters(self, hyperparams):
        """
        Log hyperparameters.
        
        Args:
            hyperparams (dict): Dictionary containing hyperparameters
        """
        # Save hyperparameters to JSON file
        hyperparams_file = os.path.join(self.experiment_dir, 'hyperparameters.json')
        with open(hyperparams_file, 'w') as f:
            json.dump(hyperparams, f, indent=4)
        
        if self.log_to_file:
            hyperparams_str = json.dumps(hyperparams, indent=4)
            self.logger.info(f"Hyperparameters:\n{hyperparams_str}")
        
        if self.use_tensorboard:
            # Log hyperparameters to TensorBoard
            self.writer.add_text('Hyperparameters', str(hyperparams))
    
    def log_model_summary(self, model, input_size=None):
        """
        Log model summary.
        
        Args:
            model (torch.nn.Module): Model to log
            input_size (tuple, optional): Input size for the model
        """
        # Count number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Log model summary
        if self.log_to_file:
            self.logger.info(f"Model: {model.__class__.__name__}")
            self.logger.info(f"Total parameters: {total_params}")
            self.logger.info(f"Trainable parameters: {trainable_params}")
        
        if self.use_tensorboard:
            # Log model graph to TensorBoard if input_size is provided
            if input_size is not None:
                try:
                    # Create dummy input
                    device = next(model.parameters()).device
                    dummy_input = torch.zeros(input_size, device=device)
                    
                    # Add model graph to TensorBoard
                    self.writer.add_graph(model, dummy_input)
                except Exception as e:
                    print(f"Failed to add model graph to TensorBoard: {e}")
    
    def log_batch(self, epoch, batch_idx, batch_size, data_size, loss, lr=None, metrics=None, prefix='Train'):
        """
        Log batch information.
        
        Args:
            epoch (int): Current epoch
            batch_idx (int): Current batch index
            batch_size (int): Batch size
            data_size (int): Total size of the dataset
            loss (float): Loss value
            lr (float, optional): Learning rate
            metrics (dict, optional): Dictionary containing metrics
            prefix (str, optional): Prefix for the log message ('Train' or 'Val')
        """
        # Compute progress
        samples_processed = min((batch_idx + 1) * batch_size, data_size)
        progress = 100.0 * (batch_idx + 1) / (data_size // batch_size + (1 if data_size % batch_size else 0))
        
        # Create log message
        log_msg = f"{prefix} Epoch: {epoch} [{samples_processed}/{data_size} ({progress:.0f}%)] Loss: {loss:.6f}"
        
        # Add learning rate to log message
        if lr is not None:
            log_msg += f" LR: {lr:.6f}"
        
        # Add metrics to log message
        if metrics is not None:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    log_msg += f" {name}: {value:.4f}"
        
        # Log message
        if self.log_to_file:
            self.logger.info(log_msg)
    
    def log_epoch(self, epoch, train_loss, val_loss, train_metrics=None, val_metrics=None, lr=None, save_model_fn=None, monitor_metric='loss'):
        """
        Log epoch information.
        
        Args:
            epoch (int): Current epoch
            train_loss (float): Training loss
            val_loss (float): Validation loss
            train_metrics (dict, optional): Dictionary containing training metrics
            val_metrics (dict, optional): Dictionary containing validation metrics
            lr (float, optional): Learning rate
            save_model_fn (callable, optional): Function to save model
            monitor_metric (str, optional): Metric to monitor for saving best model
        """
        # Update training history
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        if train_metrics:
            self.history['train_metrics'].append(train_metrics)
        if val_metrics:
            self.history['val_metrics'].append(val_metrics)
        if lr:
            self.history['learning_rate'].append(lr)
        
        # Create log message
        log_msg = f"Epoch: {epoch} Train Loss: {train_loss:.6f} Val Loss: {val_loss:.6f}"
        
        # Add learning rate to log message
        if lr is not None:
            log_msg += f" LR: {lr:.6f}"
        
        # Add metrics to log message
        if train_metrics is not None and val_metrics is not None:
            for name in train_metrics:
                if name in val_metrics and isinstance(train_metrics[name], (int, float)) and isinstance(val_metrics[name], (int, float)):
                    log_msg += f" Train {name}: {train_metrics[name]:.4f} Val {name}: {val_metrics[name]:.4f}"
        
        # Log message
        if self.log_to_file:
            self.logger.info(log_msg)
        
        # Log to TensorBoard
        if self.use_tensorboard:
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            
            if lr is not None:
                self.writer.add_scalar('LearningRate', lr, epoch)
            
            if train_metrics is not None:
                for name, value in train_metrics.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f'Metrics/{name}/train', value, epoch)
            
            if val_metrics is not None:
                for name, value in val_metrics.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f'Metrics/{name}/val', value, epoch)
        
        # Save model if it's the best one so far
        if self.save_best_model and save_model_fn is not None:
            # Determine the metric to monitor
            if monitor_metric == 'loss':
                current_val_metric = val_loss  # Lower is better
                is_better = current_val_metric < self.best_val_metric
            else:
                # For metrics like accuracy, higher is better
                if val_metrics is not None and monitor_metric in val_metrics:
                    current_val_metric = val_metrics[monitor_metric]
                    is_better = current_val_metric > self.best_val_metric
                    
                    # Update best metric (for metrics where higher is better)
                    if is_better:
                        self.best_val_metric = current_val_metric
                        self.best_epoch = epoch
                        
                        # Save best model
                        save_model_fn(os.path.join(self.experiment_dir, 'best_model.pth'))
                        
                        if self.log_to_file:
                            self.logger.info(f"New best model saved with {monitor_metric}: {current_val_metric:.6f}")
            
            # For loss and other metrics where lower is better
            if monitor_metric == 'loss' and is_better:
                self.best_val_metric = current_val_metric
                self.best_epoch = epoch
                
                # Save best model
                save_model_fn(os.path.join(self.experiment_dir, 'best_model.pth'))
                
                if self.log_to_file:
                    self.logger.info(f"New best model saved with {monitor_metric}: {current_val_metric:.6f}")
    
    def save_history(self):
        """
        Save training history to a JSON file.
        """
        # Create history file
        history_file = os.path.join(self.experiment_dir, 'history.json')
        
        # Convert history to a JSON serializable format
        serializable_history = {
            'epoch': self.history['epoch'],
            'train_loss': self.history['train_loss'],
            'val_loss': self.history['val_loss'],
            'learning_rate': self.history['learning_rate'],
            'train_metrics': [],
            'val_metrics': []
        }
        
        # Convert metrics dictionaries to JSON serializable format
        for metrics in self.history['train_metrics']:
            serializable_metrics = {}
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    serializable_metrics[name] = value
                elif isinstance(value, np.ndarray):
                    serializable_metrics[name] = value.tolist()
            serializable_history['train_metrics'].append(serializable_metrics)
        
        for metrics in self.history['val_metrics']:
            serializable_metrics = {}
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    serializable_metrics[name] = value
                elif isinstance(value, np.ndarray):
                    serializable_metrics[name] = value.tolist()
            serializable_history['val_metrics'].append(serializable_metrics)
        
        # Save history to JSON file
        with open(history_file, 'w') as f:
            json.dump(serializable_history, f, indent=4)
    
    def close(self):
        """
        Close the logger.
        """
        if self.use_tensorboard:
            self.writer.close()
        
        # Save training history
        self.save_history()
        
        if self.log_to_file:
            self.logger.info(f"Training completed. Best {self.best_val_metric:.6f} at epoch {self.best_epoch}")
            
            # Remove handlers to avoid duplicate logs
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
    
    def __del__(self):
        """
        Clean up when the logger is deleted.
        """
        self.close() 