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

from src.pipelineC.model import get_model
from src.pipelineC.dataset import get_dataloaders
from src.pipelineC.config import load_config, save_config
from src.utils.metrics import compute_segmentation_metrics
from src.utils.logging import TrainingLogger
from src.utils.visualization import visualize_point_cloud_segmentation, plot_confusion_matrix

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

class SegmentationLoss(nn.Module):
    """
    Loss function for point cloud segmentation.
    Combines cross entropy loss with optional class weighting.
    """
    def __init__(self, class_weights=None, ignore_index=-1):
        """
        Initialize segmentation loss.
        
        Args:
            class_weights (torch.Tensor, optional): Class weights for weighted loss
            ignore_index (int): Index to ignore in the loss calculation
        """
        super(SegmentationLoss, self).__init__()
        self.ignore_index = ignore_index
        self.class_weights = class_weights
        
        # Create the CrossEntropyLoss with optional class weights
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index
        )
    
    def forward(self, logits, targets):
        """
        Forward pass of the loss function.
        
        Args:
            logits (torch.Tensor): Predicted logits of shape (B, C, N)
            targets (torch.Tensor): Ground truth labels of shape (B, N)
        
        Returns:
            torch.Tensor: Loss value
        """
        # Reshape the logits to (B*N, C) and targets to (B*N)
        batch_size, num_classes, num_points = logits.size()
        logits = logits.permute(0, 2, 1).contiguous()  # (B, N, C)
        logits = logits.view(-1, num_classes)  # (B*N, C)
        targets = targets.view(-1)  # (B*N)
        
        # Compute the loss
        return self.criterion(logits, targets)

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
            inputs = data['point_features'].permute(0, 2, 1).to(device)  # (B, C, N)
            targets = data['point_labels'].to(device)  # (B, N)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)  # (B, num_classes, N)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Convert outputs to predictions
            preds = torch.argmax(outputs, dim=1)  # (B, N)
            
            # Check if model is actually learning (making varied predictions)
            # In early training, models often predict all points as the same class
            num_unique_preds = torch.unique(preds).size(0)
            if num_unique_preds == 1:
                logger.logger.debug(f"Batch {batch_idx}: Model predicting only one class: {preds[0, 0].item()}")
            
            # Store predictions and targets for metrics computation
            # Avoid storing too much data by sampling points
            sample_indices = torch.randint(0, targets.size(1), (min(targets.size(1), 1000),))
            all_preds.append(preds[:, sample_indices].cpu().numpy().flatten())
            all_targets.append(targets[:, sample_indices].cpu().numpy().flatten())
            
            # Log batch results
            if batch_idx % config['logging']['log_interval'] == 0:
                # Compute metrics for the current batch
                batch_metrics = compute_segmentation_metrics(
                    targets[:, sample_indices].flatten().cpu().numpy(),
                    preds[:, sample_indices].flatten().cpu().numpy()
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
    all_preds_np = np.concatenate(all_preds)
    all_targets_np = np.concatenate(all_targets)
    epoch_metrics = compute_segmentation_metrics(all_targets_np, all_preds_np)
    
    return epoch_loss, epoch_metrics

def validate(model, dataloader, criterion, device, epoch, logger, config, visualize=False):
    """
    Validate the model on the validation set.
    
    Args:
        model (torch.nn.Module): Model to validate
        dataloader (torch.utils.data.DataLoader): Validation data loader
        criterion (torch.nn.Module): Loss function
        device (torch.device): Device to use for validation
        epoch (int): Current epoch
        logger (TrainingLogger): Logger
        config (dict): Configuration dictionary
        visualize (bool): Whether to visualize results
    
    Returns:
        tuple: (val_loss, val_metrics)
    """
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    
    # For visualization
    vis_samples = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            # Get data
            inputs = data['point_features'].permute(0, 2, 1).to(device)  # (B, C, N)
            targets = data['point_labels'].to(device)  # (B, N)
            
            # Forward pass
            outputs = model(inputs)  # (B, num_classes, N)
            loss = criterion(outputs, targets)
            
            # Update metrics
            val_loss += loss.item()
            
            # Convert outputs to predictions
            preds = torch.argmax(outputs, dim=1)  # (B, N)
            
            # Check if model is making varied predictions
            num_unique_preds = torch.unique(preds).size(0)
            if num_unique_preds == 1:
                logger.logger.debug(f"Validation: Model predicting only one class: {preds[0, 0].item()}")
            
            # Store predictions and targets for metrics computation
            for i in range(targets.size(0)):
                all_preds.append(preds[i].cpu().numpy())
                all_targets.append(targets[i].cpu().numpy())
            
            # Store samples for visualization
            if visualize and len(vis_samples) < config['visualization'].get('num_samples', 10):
                for i in range(min(inputs.size(0), config['visualization'].get('num_samples', 10) - len(vis_samples))):
                    # Get the original point features
                    point_features = data['point_features'][i].cpu().numpy()
                    
                    # Get RGB image if available and requested
                    rgb_image = None
                    if config['visualization'].get('visualize_rgb', True):
                        if hasattr(dataloader.dataset, 'dataset') and 'rgb_image' in data:
                            rgb_image = data['rgb_image'][i].cpu().numpy() if data['rgb_image'] is not None else None
                    
                    vis_samples.append({
                        'point_features': point_features,
                        'target_labels': targets[i].cpu().numpy(),
                        'predicted_labels': preds[i].cpu().numpy(),
                        'rgb_image': rgb_image,
                        'sequence': data.get('sequence', ['unknown'])[i] if isinstance(data.get('sequence', ['unknown']), list) else 'unknown',
                        'subdir': data.get('subdir', ['unknown'])[i] if isinstance(data.get('subdir', ['unknown']), list) else 'unknown'
                    })
    
    # Compute average loss
    val_loss /= len(dataloader)
    
    # Compute metrics
    # Concatenate all predictions and targets
    flat_preds = np.concatenate([p.flatten() for p in all_preds])
    flat_targets = np.concatenate([t.flatten() for t in all_targets])
    
    val_metrics = compute_segmentation_metrics(flat_targets, flat_preds)
    
    # Log metrics
    if logger.use_tensorboard:
        for metric_name, metric_value in val_metrics.items():
            logger.writer.add_scalar(f'Val/{metric_name}', metric_value, epoch)
    
    # Create a confusion matrix
    cm = confusion_matrix(flat_targets, flat_preds, labels=[0, 1])
    
    # Log metrics in a vertical format for better readability
    logger.logger.info(f"Validation Metrics (Epoch {epoch}):")
    logger.logger.info(f"  Loss:        {val_loss:.6f}")
    logger.logger.info(f"  Accuracy:    {val_metrics['accuracy']:.4f}")
    logger.logger.info(f"  IoU (BG):    {val_metrics.get('iou_0', 0.0):.4f}")
    logger.logger.info(f"  IoU (Table): {val_metrics.get('iou_1', 0.0):.4f}")
    logger.logger.info(f"  Mean IoU:    {val_metrics['mean_iou']:.4f}")
    logger.logger.info(f"  Precision:   {val_metrics.get('precision_weighted', val_metrics.get('precision_macro', 0.0)):.4f}")
    logger.logger.info(f"  Recall:      {val_metrics.get('recall_weighted', val_metrics.get('recall_macro', 0.0)):.4f}")
    logger.logger.info(f"  F1 Score:    {val_metrics.get('f1_weighted', val_metrics.get('f1_macro', 0.0)):.4f}")
    logger.logger.info(f"  Confusion Matrix:\n{cm}")
    
    # Visualize segmentation results
    if visualize and vis_samples:
        visualization_dir = os.path.join(logger.log_dir, 'visualizations', f'epoch_{epoch}')
        os.makedirs(visualization_dir, exist_ok=True)
        
        # Visualize each sample
        for i, sample in enumerate(vis_samples):
            vis_file = os.path.join(visualization_dir, f'sample_{i}.png')
            visualize_point_cloud_segmentation(
                point_cloud=sample['point_features'][:, :3],  # XYZ coordinates
                point_colors=sample['point_features'][:, 3:6] if sample['point_features'].shape[1] >= 6 else None,  # RGB colors if available
                target_labels=sample['target_labels'],
                pred_labels=sample['predicted_labels'],
                rgb_image=sample['rgb_image'],
                title=f"Sample {i} - {sample['sequence']}/{sample['subdir']}",
                save_path=vis_file
            )
            
            logger.logger.info(f"Saved visualization to {vis_file}")
    
    return val_loss, val_metrics

def main(config_file):
    """
    Main function for training Pipeline C.
    
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
    
    # Create directories
    experiment_dir = os.path.join(config['logging']['log_dir'], experiment_name)
    checkpoint_dir = os.path.join(config['logging']['checkpoint_dir'], experiment_name)
    
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create tensorboard directory
    tensorboard_dir = os.path.join('results/pipelineC/tb_logs')
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
    
    # Get class weights from the training set
    if 'use_weighted_loss' in config['loss'] and config['loss']['use_weighted_loss']:
        if 'class_weights' in config['loss']:
            class_weights = torch.tensor(config['loss']['class_weights'], dtype=torch.float32).to(device)
            logger.logger.info(f"Using predefined class weights: {config['loss']['class_weights']}")
        else:
            # Try to extract class weights from the dataset
            try:
                dataset = train_loader.dataset
                # Check if dataset is a Subset
                if isinstance(dataset, torch.utils.data.Subset):
                    class_weights = dataset.dataset.get_class_weights().to(device)
                else:
                    class_weights = dataset.get_class_weights().to(device)
                logger.logger.info(f"Using class weights from dataset: {class_weights.cpu().numpy()}")
            except (AttributeError, NotImplementedError):
                logger.logger.warning("Could not extract class weights from dataset. Using default weights.")
                class_weights = torch.tensor([1.0, 2.0], dtype=torch.float32).to(device)
    else:
        class_weights = None
    
    logger.logger.info(f"Dataset loaded: {len(train_loader.dataset)} training, "
                  f"{len(val_loader.dataset)} validation, {len(test_loader.dataset)} test samples")
    
    # Get model
    model = get_model(config)
    model = model.to(device)
    
    # Log model summary
    logger.log_model_summary(model)
    
    # Define loss function
    criterion = SegmentationLoss(class_weights=class_weights)
    
    # Create optimizer (Adam only)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler (StepLR only)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training']['lr_decay_step'],
        gamma=config['training']['lr_decay']
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_val_miou = 0.0
    patience = config['training']['early_stopping']
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
            logger=logger,
            config=config,
            visualize=(epoch % 10 == 0)  # Visualize every 10 epochs
        )
        
        # Update learning rate
        scheduler.step()
        
        # Print train and val loss on the same line for comparison
        logger.logger.info(f"Epoch {epoch}/{config['training']['epochs']} - "
                     f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                     f"Mean IoU: {val_metrics['mean_iou']:.4f}, "
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
        
        # Check for best model by mean IoU
        if val_metrics['mean_iou'] > best_val_miou:
            best_val_miou = val_metrics['mean_iou']
            # Save best model by mean IoU
            best_miou_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
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
            }, best_miou_checkpoint_path)
            logger.logger.info(f"New best model (mean IoU) saved at epoch {epoch}")
        
        # Early stopping
        if early_stopping_counter >= patience:
            logger.logger.info(f"Early stopping triggered after {patience} epochs without improvement")
            break
    
    # Close logger
    logger.close()
    
    logger.logger.info("Training completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Pipeline C: Point Cloud Segmentation')
    parser.add_argument('--config', type=str, default='src/pipelineC/config.yaml', help='Path to configuration file')
    args = parser.parse_args()
    
    main(args.config) 