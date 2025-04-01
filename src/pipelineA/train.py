import os
import sys
import json
import time
import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import yaml
import random
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from src.pipelineA.model import get_model
from src.pipelineA.dataset import get_dataloaders
from src.pipelineA.config import load_config, save_config
from src.utils.metrics import compute_metrics_from_logits, compute_classification_metrics
from src.utils.logging import TrainingLogger

def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_one_epoch(model, train_loader, criterion, optimizer, device, logger, epoch, config):
    """
    Train the model for one epoch.
    
    Args:
        model (torch.nn.Module): Model to train
        train_loader (torch.utils.data.DataLoader): Training data loader
        criterion (torch.nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to use for training
        logger (TrainingLogger): Logger
        epoch (int): Current epoch
        config (dict): Configuration dictionary
    
    Returns:
        tuple: (epoch_loss, epoch_metrics)
    """
    model.train()
    epoch_loss = 0.0
    all_preds = []
    all_targets = []
    
    # Enable gradient computation
    with torch.set_grad_enabled(True):
        for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            # Get data
            inputs = data['point_cloud'].to(device)
            targets = data['label'].to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Convert outputs to predictions
            preds = torch.argmax(outputs, dim=1)
            
            # Store predictions and targets for metrics computation
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Log batch results
            if batch_idx % config['logging']['log_interval'] == 0:
                # Compute metrics for the current batch
                batch_metrics = compute_metrics_from_logits(
                    outputs, 
                    targets,
                    task='classification'
                )
                
                # Log batch
                logger.log_batch(
                    epoch=epoch,
                    batch_idx=batch_idx,
                    batch_size=inputs.size(0),
                    data_size=len(train_loader.dataset),
                    loss=loss.item(),
                    lr=optimizer.param_groups[0]['lr'],
                    metrics=batch_metrics,
                    prefix='Train'
                )
    
    # Compute average loss
    epoch_loss /= len(train_loader)
    
    # Compute metrics for the entire epoch
    if all_preds and all_targets:
        all_preds_np = np.array(all_preds)
        all_targets_np = np.array(all_targets)
        epoch_metrics = compute_classification_metrics(all_targets_np, all_preds_np)
    else:
        # If no predictions were made (all batches failed), return empty metrics
        epoch_metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'specificity': 0.0
        }
    
    return epoch_loss, epoch_metrics

def validate(model, dataloader, criterion, device, epoch, logger):
    """
    Validate the model on the validation set.
    
    Args:
        model (torch.nn.Module): Model to validate
        dataloader (torch.utils.data.DataLoader): Validation data loader
        criterion (torch.nn.Module): Loss function
        device (torch.device): Device to use for validation
        epoch (int): Current epoch
        logger (TrainingLogger): Logger
    
    Returns:
        tuple: (val_loss, val_metrics)
    """
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            # Get data
            inputs = data['point_cloud'].to(device)
            targets = data['label'].to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Update metrics
            val_loss += loss.item()
            
            # Convert outputs to predictions
            preds = torch.argmax(outputs, dim=1)
            
            # Store predictions and targets for metrics computation
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Compute average loss
    val_loss /= len(dataloader)
    
    # Compute metrics
    val_metrics = compute_classification_metrics(
        np.array(all_targets),
        np.array(all_preds)
    )
    
    # Log metrics to tensorboard
    if logger.use_tensorboard:
        for metric_name, metric_value in val_metrics.items():
            logger.writer.add_scalar(f'Val/{metric_name}', metric_value, epoch)
    
    # Create a confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Log metrics in a vertical format for better readability
    logger.logger.info(f"Validation Metrics (Epoch {epoch}):")
    logger.logger.info(f"  Loss:        {val_loss:.6f}")
    logger.logger.info(f"  Accuracy:    {val_metrics['accuracy']:.4f}")
    logger.logger.info(f"  Precision:   {val_metrics['precision']:.4f}")
    logger.logger.info(f"  Recall:      {val_metrics['recall']:.4f}")
    logger.logger.info(f"  F1 Score:    {val_metrics['f1_score']:.4f}")
    logger.logger.info(f"  Specificity: {val_metrics['specificity']:.4f}")
    logger.logger.info(f"  Confusion Matrix:\n{cm}")
    
    return val_loss, val_metrics

def main(config_file):
    """
    Main function for training Pipeline A.
    
    Args:
        config_file (str): Path to the YAML configuration file
    """
    # Load configuration
    config = load_config(config_file)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create timestamp for experiment
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config['logging']['experiment_name']}_{timestamp}"
    
    # Create directories with timestamp
    experiment_dir = os.path.join(config['logging']['log_dir'], experiment_name)
    checkpoint_dir = os.path.join(config['logging']['checkpoint_dir'], experiment_name)
    
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create fixed tensorboard directory
    tensorboard_dir = os.path.join('results/pipelineA/tb_logs')
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # Update config with experiment name
    config['logging']['experiment_dir'] = experiment_dir
    config['logging']['timestamp'] = timestamp
    
    # Save the config
    save_config(config, os.path.join(experiment_dir, 'config.json'))
    
    # Initialize logger
    logger = TrainingLogger(
        log_dir=experiment_dir,
        experiment_name=experiment_name,
        use_tensorboard=config['logging']['use_tensorboard'],
        tensorboard_dir=tensorboard_dir
    )
    
    # Log start of training
    logger.logger.info(f"Starting training experiment: {experiment_name}")
    logger.logger.info(f"Using device: {device}")
    
    # Log hyperparameters
    logger.log_hyperparameters(config)
    
    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config_file)
    logger.logger.info(f"Dataset loaded: {len(train_loader.dataset)} training, "
                  f"{len(val_loader.dataset)} validation, {len(test_loader.dataset)} test samples")
    
    # Get model
    model = get_model(config)
    model = model.to(device)
    
    # Log model summary
    logger.log_model_summary(model, input_size=(1, config['data']['num_points'], 3))
    
    # Define loss function, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizer (Adam only)
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config['training']['lr']),
        weight_decay=float(config['training']['weight_decay'])
    )
    
    # Create scheduler (StepLR only)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(config['training']['lr_decay_step']),
        gamma=float(config['training']['lr_decay'])
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    patience = int(config['training']['early_stopping'])
    early_stopping_counter = 0
    
    for epoch in range(1, config['training']['epochs'] + 1):
        # Train for one epoch
        train_loss, train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            logger=logger,
            epoch=epoch,
            config=config
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            logger=logger
        )
        
        # Update learning rate
        scheduler.step()
        
        # Print train and val loss on the same line for comparison
        logger.logger.info(f"Epoch {epoch}/{config['training']['epochs']} - "
                     f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                     f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Log epoch results
        logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            lr=scheduler.get_last_lr()[0]
        )
        
        # Save the model checkpoint (latest model)
        latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': config
        }, latest_checkpoint_path)
        
        # Check for best model by validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model by loss
            best_loss_checkpoint_path = os.path.join(checkpoint_dir, 'best_loss_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'config': config
            }, best_loss_checkpoint_path)
            logger.logger.info(f"New best model (loss) saved at epoch {epoch}")
            
            # Reset early stopping counter
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        # Check for best model by F1 score
        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            # Save best model by F1 score
            best_f1_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'config': config
            }, best_f1_checkpoint_path)
            logger.logger.info(f"New best model (F1 score) saved at epoch {epoch}")
        
        # Early stopping
        if early_stopping_counter >= patience:
            logger.logger.info(f"Early stopping triggered after {patience} epochs without improvement")
            break
    
    # Close logger
    logger.close()
    
    logger.logger.info("Training completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Pipeline A: Point Cloud Classification')
    parser.add_argument('--config', type=str, default='src/pipelineA/config.yaml', help='Path to configuration file')
    args = parser.parse_args()
    
    main(args.config) 