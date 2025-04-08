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
import torch.nn.functional as F

from src.pipelineC.model import get_model
from src.pipelineC.dataset import get_dataloaders
from src.pipelineC.config import load_config, save_config
from src.utils.metrics import compute_segmentation_metrics
from src.utils.logging import TrainingLogger
from src.utils.visualization import visualize_point_cloud_segmentation, plot_confusion_matrix
from src.pipelineC.eval import evaluate

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

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for segmentation.
    Focal loss addresses class imbalance by down-weighting easy examples.
    
    FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)
    where p_t is the predicted probability for the correct class.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_index=-1):
        """
        Initialize focal loss.
        
        Args:
            alpha (torch.Tensor, optional): Weight for each class
            gamma (float): Focusing parameter (>= 0). Larger gamma focuses more on hard examples
            reduction (str): 'mean', 'sum', or 'none'
            ignore_index (int): Index to ignore in the loss calculation
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.ce_loss = None  # Will initialize in forward pass
        
    def forward(self, logits, targets):
        """
        Forward pass of focal loss.
        
        Args:
            logits (torch.Tensor): Predicted logits of shape (B, C, N) or (B*N, C)
            targets (torch.Tensor): Ground truth labels of shape (B, N) or (B*N)
        
        Returns:
            torch.Tensor: Loss value
        """
        # Get device
        device = logits.device
        
        # Move alpha to the same device if needed
        if self.alpha is not None and self.alpha.device != device:
            self.alpha = self.alpha.to(device)
        
        # Initialize CE loss if not already done
        if self.ce_loss is None or (self.alpha is not None and self.alpha.device != device):
            self.ce_loss = nn.CrossEntropyLoss(
                weight=self.alpha, 
                reduction='none',
                ignore_index=self.ignore_index
            )
        
        # Ensure inputs are properly shaped
        if logits.dim() == 3:  # (B, C, N)
            batch_size, num_classes, num_points = logits.size()
            logits = logits.permute(0, 2, 1).contiguous()  # (B, N, C)
            logits = logits.view(-1, num_classes)  # (B*N, C)
            targets = targets.view(-1)  # (B*N)
            
        # Compute cross entropy loss
        ce = self.ce_loss(logits, targets)
        
        # Compute probabilities with softmax
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # Probability for target class
        
        # Compute focal weight
        focal_weight = (1 - pt).pow(self.gamma)
        
        # Compute focal loss
        loss = focal_weight * ce
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class DiceLoss(nn.Module):
    """
    Dice Loss implementation for segmentation.
    Dice loss optimizes the Dice coefficient (F1 score) directly.
    """
    def __init__(self, smooth=1.0, ignore_index=-1):
        """
        Initialize dice loss.
        
        Args:
            smooth (float): Smoothing term to avoid division by zero
            ignore_index (int): Index to ignore in the loss calculation
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        
    def forward(self, logits, targets):
        """
        Forward pass of dice loss.
        
        Args:
            logits (torch.Tensor): Predicted logits of shape (B, C, N) or (B*N, C)
            targets (torch.Tensor): Ground truth labels of shape (B, N) or (B*N)
        
        Returns:
            torch.Tensor: Loss value
        """
        # Ensure inputs are properly shaped
        if logits.dim() == 3:  # (B, C, N)
            batch_size, num_classes, num_points = logits.size()
            logits = logits.permute(0, 2, 1).contiguous()  # (B, N, C)
            logits = logits.view(-1, num_classes)  # (B*N, C)
            targets = targets.view(-1)  # (B*N)
        
        # Create one-hot encoding for targets
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Mask out ignored indices
        mask = (targets != self.ignore_index).float().unsqueeze(1)
        one_hot = one_hot * mask
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)
        
        # Multiply by mask to ignore certain indices
        probs = probs * mask
        
        # Calculate dice coefficient for each class
        numerator = 2 * (probs * one_hot).sum(0) + self.smooth
        denominator = probs.sum(0) + one_hot.sum(0) + self.smooth
        dice = numerator / denominator
        
        # Calculate mean dice loss
        loss = 1 - dice.mean()
        
        return loss

class CombinedLoss(nn.Module):
    """
    Combines different loss functions with optional weights.
    """
    def __init__(self, losses, weights=None):
        """
        Initialize combined loss.
        
        Args:
            losses (list): List of loss functions
            weights (list, optional): Weights for each loss
        """
        super(CombinedLoss, self).__init__()
        self.losses = losses
        self.weights = weights if weights is not None else [1.0] * len(losses)
        
    def forward(self, logits, targets):
        """
        Forward pass of combined loss.
        
        Args:
            logits (torch.Tensor): Predicted logits
            targets (torch.Tensor): Ground truth labels
        
        Returns:
            torch.Tensor: Combined loss value
        """
        combined_loss = 0.0
        for i, loss_fn in enumerate(self.losses):
            if isinstance(loss_fn, nn.CrossEntropyLoss):  # ce
                batch_size, num_classes, num_points = logits.size()
                logits = logits.permute(0, 2, 1).contiguous()  # (B, N, C)
                logits = logits.view(-1, num_classes)  # (B*N, C)
                targets = targets.view(-1)  # (B*N)
            combined_loss += self.weights[i] * loss_fn(logits, targets)
        return combined_loss

class SegmentationLoss(nn.Module):
    """
    Loss function for point cloud segmentation.
    Combines different loss functions including cross entropy, focal loss, and dice loss.
    """
    def __init__(self, class_weights=None, ignore_index=-1, loss_type='ce', focal_gamma=2.0, loss_weights=None):
        """
        Initialize segmentation loss.
        
        Args:
            class_weights (list or torch.Tensor, optional): Class weights for weighted loss
            ignore_index (int): Index to ignore in the loss calculation
            loss_type (str): Type of loss to use ('ce', 'focal', 'dice', 'combined')
            focal_gamma (float): Gamma parameter for focal loss
            loss_weights (list, optional): Weights for combined loss
        """
        super(SegmentationLoss, self).__init__()
        self.ignore_index = ignore_index
        self.loss_type = loss_type.lower()
        
        # Convert class_weights to tensor if it's a list
        if class_weights is not None:
            if isinstance(class_weights, list):
                self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
            else:
                self.class_weights = class_weights
        else:
            self.class_weights = None
            
        # Note: We'll initialize the actual criterion in the forward method
        # to ensure it's on the correct device
        self.criterion = None
        self.focal_gamma = focal_gamma
        self.loss_weights = loss_weights
    
    def forward(self, logits, targets):
        """
        Forward pass of the loss function.
        
        Args:
            logits (torch.Tensor): Predicted logits of shape (B, C, N)
            targets (torch.Tensor): Ground truth labels of shape (B, N)
        
        Returns:
            torch.Tensor: Loss value
        """
        # Get device of input tensors
        device = logits.device
        
        # Move class_weights to the same device if needed
        if self.class_weights is not None and self.class_weights.device != device:
            self.class_weights = self.class_weights.to(device)
            
        # Create the loss function on first forward pass or if device changes
        if self.criterion is None:
            if self.loss_type == 'ce':
                self.criterion = nn.CrossEntropyLoss(
                    weight=self.class_weights,
                    ignore_index=self.ignore_index
                )
            elif self.loss_type == 'focal':
                self.criterion = FocalLoss(
                    alpha=self.class_weights,
                    gamma=self.focal_gamma,
                    ignore_index=self.ignore_index
                )
            elif self.loss_type == 'dice':
                self.criterion = DiceLoss(
                    ignore_index=self.ignore_index
                )
            elif self.loss_type == 'combined':
                # Default: equally weighted CE and Dice loss
                weights = self.loss_weights if self.loss_weights else [0.5, 0.5]
                self.criterion = CombinedLoss(
                    losses=[
                        nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=self.ignore_index),
                        DiceLoss(ignore_index=self.ignore_index)
                    ],
                    weights=weights
                )
        
        # For CrossEntropyLoss, we need to reshape 
        if self.loss_type == 'ce':
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
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        criterion (nn.Module): Loss function
        optimizer (Optimizer): Optimizer
        device (torch.device): Device to use
        logger (TrainingLogger): Logger object
        epoch (int): Current epoch
        config (dict): Configuration dictionary
    
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Train]")
    
    for batch_idx, data in enumerate(pbar):
        # Get data
        inputs = data['point_features'].to(device)
        targets = data['point_labels'].to(device)
        counts = data['label_counts']
        
        class_weight = torch.sum(counts, dim=0)
        class_weight = torch.max(class_weight) / (class_weight + 1e-9)
        # print(f"class weight: {class_weight}")
        
        # Transpose input if needed - model expects (B, C, N) but data might be (B, N, C)
        # if inputs.shape[1] == config['data']['num_points']:  # If second dimension is num_points, we need to transpose
        inputs = inputs.transpose(1, 2)  # (B, N, C) -> (B, C, N)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        criterion.class_weights = class_weight
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Update loss
        total_loss += loss.item()
        
        # Calculate predictions
        preds = torch.argmax(outputs, dim=1)
        
        # Store predictions and targets for metrics calculation
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
        })
        
        # Log batch metrics
        if batch_idx % config['logging']['log_interval'] == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logger.log_batch(
                epoch=epoch,
                batch_idx=batch_idx,
                batch_size=len(inputs),
                data_size=len(train_loader.dataset),
                loss=loss.item(),
                lr=current_lr,
                prefix='Train'
            )
    
    # Compute average loss
    train_loss = total_loss / len(train_loader)
    
    # Compute metrics
    # Concatenate all predictions and targets
    flat_preds = np.concatenate([p.flatten() for p in all_preds])
    flat_targets = np.concatenate([t.flatten() for t in all_targets])
    
    train_metrics = compute_segmentation_metrics(flat_targets, flat_preds)
    
    # Log metrics
    if logger.use_tensorboard:
        for metric_name, metric_value in train_metrics.items():
            logger.writer.add_scalar(f'Train/{metric_name}', metric_value, epoch)
    
    # Print summary to console
    print(f"\nTrain Epoch: {epoch+1} Summary:")
    print(f"  Loss:            {train_loss:.4f}")
    print(f"  Accuracy:        {train_metrics['accuracy']:.4f}")
    print(f"  Mean IoU:        {train_metrics['mean_iou']:.4f}")
    print(f"  Background IoU:  {train_metrics['iou_background']:.4f}")
    print(f"  Table IoU:       {train_metrics['iou_table']:.4f}")
    print(f"  F1 Score:        {train_metrics.get('f1_weighted', 0.0):.4f}")
    
    return train_loss, train_metrics

def validate(model, dataloader, criterion, device, epoch, logger, config, visualize=False):
    """
    Validate the model on the validation set.
    
    Args:
        model (nn.Module): Model to validate
        dataloader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        device (torch.device): Device to use
        epoch (int): Current epoch
        logger (TrainingLogger): Logger object
        config (dict): Configuration dictionary
        visualize (bool): Whether to visualize predictions
    
    Returns:
        tuple: (average validation loss, validation metrics)
    """
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    vis_samples = []
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Val]")
    
    with torch.no_grad():
        for batch_idx, data in enumerate(pbar):
            # Get data
            inputs = data['point_features'].to(device)
            targets = data['point_labels'].to(device)
            counts = data['label_counts']
        
            class_weight = torch.sum(counts, dim=0)
            class_weight = torch.max(class_weight) / (class_weight + 1e-9)
            # print(f"class weight: {class_weight}")
            
            # Transpose input if needed - model expects (B, C, N) but data might be (B, N, C)
            if inputs.shape[1] == config['data']['num_points']:  # If second dimension is num_points, we need to transpose
                inputs = inputs.transpose(1, 2)  # (B, N, C) -> (B, C, N)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            criterion.class_weights = class_weight
            loss = criterion(outputs, targets)
            
            # Update loss
            val_loss += loss.item()
            
            # Calculate predictions
            preds = torch.argmax(outputs, dim=1)
            
            # Store predictions and targets for metrics calculation
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'val_loss': f"{loss.item():.4f}",
                'avg_val_loss': f"{val_loss / (batch_idx + 1):.4f}"
            })
            
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
    logger.logger.info(f"  Loss:            {val_loss:.6f}")
    logger.logger.info(f"  Accuracy:        {val_metrics['accuracy']:.4f}")
    logger.logger.info(f"  IoU (Background): {val_metrics.get('iou_background', 0.0):.4f}")
    logger.logger.info(f"  IoU (Table):     {val_metrics.get('iou_table', 0.0):.4f}")
    logger.logger.info(f"  Mean IoU:        {val_metrics['mean_iou']:.4f}")
    logger.logger.info(f"  F1 Score:        {val_metrics.get('f1_weighted', val_metrics.get('f1_macro', 0.0)):.4f}")
    logger.logger.info(f"  Confusion Matrix:\n{cm}")
    
    # Print summary to console for better visibility
    print(f"\nValidation Epoch: {epoch+1} Summary:")
    print(f"  Loss:            {val_loss:.4f}")
    print(f"  Accuracy:        {val_metrics['accuracy']:.4f}")
    print(f"  Mean IoU:        {val_metrics['mean_iou']:.4f}")
    print(f"  Background IoU:  {val_metrics['iou_background']:.4f}")
    print(f"  Table IoU:       {val_metrics['iou_table']:.4f}")
    print(f"  F1 Score:        {val_metrics.get('f1_weighted', 0.0):.4f}")
    
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
        config_file (str): Path to the configuration file
    """
    # Load configuration
    config = load_config(config_file)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config_file)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Get class weights from dataset if available
    # class_weights = torch.zeros((1, 2))
    # pbar = tqdm(train_loader, desc=f"[Calculating weights]")
    # for batch_idx, data in enumerate(pbar):
    #     counts = data['label_counts']
    #     class_weights += torch.sum(counts, dim=0)
    # class_weights = torch.max(class_weights) / (class_weights + 1e-9)
    # class_weights = config['loss'].get('class_weights', None)
    # print(f"class weights: {class_weights}")
    
    # Create model
    model = get_model(config)
    model = model.to(device)
    
    # Create criterion with the configured loss type
    loss_type = config['loss'].get('loss_type', 'ce')
    focal_gamma = config['loss'].get('focal_gamma', 2.0)
    loss_weights = config['loss'].get('loss_weights', None)
    
    if config['loss'].get('use_weighted_loss', False) and class_weights is not None:
        print(f"Using weighted {loss_type} loss with class weights: {class_weights}")
        criterion = SegmentationLoss(
            class_weights=class_weights,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            loss_weights=loss_weights
        )
    else:
        print(f"Using {loss_type} loss")
        criterion = SegmentationLoss(
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            loss_weights=loss_weights
        )
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training']['lr_decay_step'],
        gamma=config['training']['lr_decay']
    )
    
    # Create experiment directory
    experiment_dir = os.path.join(
        config['logging']['log_dir'],
        config['logging']['experiment_name']
    )
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save configuration
    save_config(config, os.path.join(experiment_dir, 'config.json'))
    
    # Create logger
    logger = TrainingLogger(
        log_dir=experiment_dir,
        experiment_name=config['logging']['experiment_name'],
        use_tensorboard=config['logging']['use_tensorboard'],
        log_to_file=True,
        save_best_model=config['logging']['save_checkpoint']
    )
    
    # Log hyperparameters
    logger.log_hyperparameters(config)
    
    # Log model summary
    logger.log_model_summary(model)
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(
        config['logging']['checkpoint_dir'],
        config['logging']['experiment_name']
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Add custom scalars to tensorboard
    if logger.use_tensorboard:
        logger.writer.add_custom_scalars({
            'IoU': {
                'Background vs Table': ['Multiline', ['Val/iou_background', 'Val/iou_table']],
            },
            'Metrics': {
                'Train vs Val Loss': ['Multiline', ['Train/loss', 'Val/loss']],
                'Train vs Val Accuracy': ['Multiline', ['Train/accuracy', 'Val/accuracy']],
                'Train vs Val Mean IoU': ['Multiline', ['Train/mean_iou', 'Val/mean_iou']],
            }
        })
    
    # Training loop
    best_val_loss = float('inf')
    best_val_iou = 0.0
    best_epoch = 0
    early_stopping_counter = 0
    
    for epoch in range(config['training']['epochs']):
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
        
        # Update best model
        current_val_iou = val_metrics['mean_iou']
        if current_val_iou > best_val_iou:
            best_val_iou = current_val_iou
            best_val_loss = val_loss
            best_epoch = epoch
            
            # Save best model
            if config['logging']['save_checkpoint']:
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'metrics': val_metrics,
                    'config': config
                }, checkpoint_path)
                print(f"Best model saved at epoch {epoch+1} with IoU: {best_val_iou:.4f}")
            
            # Reset early stopping counter
            early_stopping_counter = 0
        else:
            # Increment early stopping counter
            # early_stopping_counter += 1
            
            # Check if early stopping criteria is met
            if early_stopping_counter >= config['training']['early_stopping']:
                print(f"Early stopping at epoch {epoch+1}. Best IoU: {best_val_iou:.4f} at epoch {best_epoch+1}")
                break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 and config['logging']['save_checkpoint']:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'metrics': val_metrics,
                'config': config
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")
    
    # Train finished
    print("\nTraining finished!")
    print(f"Best validation performance at epoch {best_epoch+1}:")
    print(f"  IoU (Table): {best_val_iou:.4f}")
    print(f"  Loss: {best_val_loss:.4f}")
    
    # Load best model for testing
    if config['logging']['save_checkpoint']:
        checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
        
    # Test best model
    test_output_dir = os.path.join(experiment_dir, 'test_results')
    test_loss, test_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        output_dir=test_output_dir,
        visualize=True,
        num_vis_samples=config['visualization'].get('num_samples', 10)
    )
    
    # Print final test results
    print("\nFinal Test Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Mean IoU: {test_metrics['mean_iou']:.4f}")
    print(f"  Background IoU: {test_metrics['iou_background']:.4f}")
    print(f"  Table IoU: {test_metrics['iou_table']:.4f}")
    print(f"  F1 Score: {test_metrics.get('f1_weighted', 0.0):.4f}")
    
    # Close logger
    logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Pipeline C: Point Cloud Segmentation')
    parser.add_argument('--config', type=str, default='src/pipelineC/config.yaml', help='Path to configuration file')
    args = parser.parse_args()
    
    main(args.config) 