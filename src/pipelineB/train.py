import os
import sys
import json
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import yaml
import random
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.pipelineB.model import get_model
from src.pipelineB.dataset import get_dataloaders
from src.pipelineB.config import load_config, save_config
from src.utils.metrics import compute_metrics_from_logits, compute_classification_metrics
from src.utils.logging import TrainingLogger
from src.utils.visualization import visualize_classification_results, plot_confusion_matrix
from src.utils.depth_losses import DepthEstimationLoss

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
        criterion (torch.nn.Module): Classification loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to use for training
        logger (TrainingLogger): Logger
        epoch (int): Current epoch
        config (dict): Configuration dictionary
    
    Returns:
        tuple: (epoch_loss, epoch_metrics)
    """
    # Enable anomaly detection to pinpoint the in-place operation issue
    # torch.autograd.set_detect_anomaly(True)
    
    model.train()
    epoch_loss = 0.0
    all_preds = []
    all_targets = []
    
    # Initialize depth loss if depth estimator is not frozen
    depth_loss_fn = None
    depth_loss_weight = 0.0
    
    if not model.freeze_depth_estimator:
        depth_loss_weight = config['training'].get('depth_loss_weight', 0.0)
        if depth_loss_weight > 0:
            si_weight = config['training'].get('si_weight', 1.0)
            edge_weight = config['training'].get('edge_weight', 0.1)
            depth_loss_fn = DepthEstimationLoss(
                si_weight=si_weight,
                edge_weight=edge_weight
            )
    
    # Enable gradient computation
    with torch.set_grad_enabled(True):
        for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            # Get data
            inputs = data['rgb_image'].to(device)
            targets = data['label'].to(device)
            
            # Get ground truth depth if available and depth loss is enabled
            gt_depth = None
            if not model.freeze_depth_estimator and depth_loss_weight > 0 and 'gt_depth' in data and data['gt_depth'] is not None:
                gt_depth = data['gt_depth'].to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass - get depth maps only if needed for depth loss
            if not model.freeze_depth_estimator and gt_depth is not None and depth_loss_fn is not None:
                outputs, pred_depth = model(inputs, return_depth=True)
                
                # Compute classification loss
                cls_loss = criterion(outputs, targets)
                
                # Compute depth loss
                depth_loss, depth_loss_dict = depth_loss_fn(pred_depth, gt_depth)
                
                # Combine losses
                loss = cls_loss + depth_loss_weight * depth_loss
                
                # Log depth loss components
                if batch_idx % config['logging']['log_interval'] == 0:
                    logger.logger.info(f"Batch {batch_idx}: Depth SI Loss: {depth_loss_dict['si_loss']:.4f}, "
                                     f"Edge Loss: {depth_loss_dict['edge_loss']:.4f}")
            else:
                # Just get classification outputs if no depth loss
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
                
                # Log to tensorboard
                if logger.use_tensorboard:
                    # Log loss
                    logger.writer.add_scalar(
                        'Batch/Loss',
                        loss.item(),
                        epoch * len(train_loader) + batch_idx
                    )
                    
                    # Log metrics
                    for metric_name, metric_value in batch_metrics.items():
                        logger.writer.add_scalar(
                            f'Batch/{metric_name}',
                            metric_value,
                            epoch * len(train_loader) + batch_idx
                        )
    
    # Compute average loss
    epoch_loss /= len(train_loader)
    
    # Compute epoch metrics
    epoch_metrics = compute_classification_metrics(
        np.array(all_targets),
        np.array(all_preds)
    )
    
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
        tuple: (val_loss, val_metrics, depth_metrics)
    """
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    
    # For depth evaluation
    depth_errors = {
        'rmse': [],
        'mae': [],
        'rel': [],
        'a1': [],
        'a2': [],
        'a3': []
    }
    valid_depth_samples = 0
    
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            # Get data
            inputs = data['rgb_image'].to(device)
            targets = data['label'].to(device)
            
            # Get ground truth depth if available
            gt_depth = data.get('gt_depth')
            if gt_depth is not None:
                gt_depth = gt_depth.to(device)
            
            # Forward pass with depth map retrieval
            outputs, pred_depth = model(inputs, return_depth=True)
            loss = criterion(outputs, targets)
            
            # Update metrics
            val_loss += loss.item()
            
            # Convert outputs to predictions
            preds = torch.argmax(outputs, dim=1)
            
            # Store predictions and targets for metrics computation
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Evaluate depth estimation if ground truth is available
            if gt_depth is not None:
                for i in range(len(inputs)):
                    sample_gt_depth = gt_depth[i]
                    sample_pred_depth = pred_depth[i]
                    
                    # Skip samples with no valid depth
                    if torch.sum(sample_gt_depth > 0) < 100:  # Require at least 100 valid pixels
                        continue
                    
                    # Compute depth metrics
                    from src.utils.metrics import compute_depth_metrics
                    metrics = compute_depth_metrics(sample_pred_depth, sample_gt_depth)
                    
                    # Skip samples with invalid metrics
                    if any(np.isnan(list(metrics.values()))):
                        continue
                    
                    # Accumulate metrics
                    for k, v in metrics.items():
                        depth_errors[k].append(v)
                    
                    valid_depth_samples += 1
    
    # Compute average loss
    val_loss /= len(dataloader)
    
    # Compute classification metrics
    val_metrics = compute_classification_metrics(
        np.array(all_targets),
        np.array(all_preds)
    )
    
    # Compute average depth metrics
    depth_metrics = {}
    if valid_depth_samples > 0:
        for k, v in depth_errors.items():
            if v:  # Check if list is not empty
                depth_metrics[k] = float(np.mean(v))
            else:
                depth_metrics[k] = float('nan')
    
    # Log metrics to tensorboard
    if logger.use_tensorboard:
        for metric_name, metric_value in val_metrics.items():
            logger.writer.add_scalar(f'Val/{metric_name}', metric_value, epoch)
        
        # Log depth metrics
        for metric_name, metric_value in depth_metrics.items():
            if not np.isnan(metric_value):
                logger.writer.add_scalar(f'Val/Depth_{metric_name}', metric_value, epoch)
    
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
    
    # Log depth metrics if available
    if depth_metrics:
        logger.logger.info(f"Depth Estimation Metrics (Epoch {epoch}):")
        logger.logger.info(f"  Valid Samples: {valid_depth_samples}")
        logger.logger.info(f"  RMSE:  {depth_metrics.get('rmse', float('nan')):.4f} meters")
        logger.logger.info(f"  MAE:   {depth_metrics.get('mae', float('nan')):.4f} meters")
        logger.logger.info(f"  REL:   {depth_metrics.get('rel', float('nan')):.4f}")
        logger.logger.info(f"  δ1:    {depth_metrics.get('a1', float('nan')):.4f} (% under 1.25)")
        logger.logger.info(f"  δ2:    {depth_metrics.get('a2', float('nan')):.4f} (% under 1.25²)")
        logger.logger.info(f"  δ3:    {depth_metrics.get('a3', float('nan')):.4f} (% under 1.25³)")
    
    return val_loss, val_metrics, depth_metrics

def save_depth_visualization(pred_depth, gt_depth, image_idx, output_dir, epoch=None, batch_idx=None):
    """
    保存深度图的可视化比较
    
    Args:
        pred_depth (torch.Tensor): 预测的深度图
        gt_depth (torch.Tensor): 真实的深度图
        image_idx (int): 样本索引
        output_dir (str): 保存目录
        epoch (int, optional): 当前轮次
        batch_idx (int, optional): 批次索引
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 将张量转换为NumPy数组
    pred_depth_np = pred_depth.detach().cpu().numpy()
    gt_depth_np = gt_depth.detach().cpu().numpy()
    
    # 创建可视化
    plt.figure(figsize=(12, 5))
    
    # 预测深度图
    plt.subplot(121)
    plt.title("MiDaS Predicted Depth")
    plt.imshow(pred_depth_np, cmap='plasma')
    plt.colorbar(label='Depth')
    
    # 真实深度图
    plt.subplot(122)
    plt.title("Ground Truth Depth")
    plt.imshow(gt_depth_np, cmap='plasma')
    plt.colorbar(label='Depth')
    
    # 创建文件名
    if epoch is not None and batch_idx is not None:
        filename = f"depth_vis_e{epoch}_b{batch_idx}_s{image_idx}.png"
    else:
        filename = f"depth_vis_sample_{image_idx}.png"
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()

def main(config_file):
    """
    Main function for training Pipeline B.
    
    Args:
        config_file (str): Path to configuration file
    """
    # Load configuration
    config = load_config(config_file)
    
    # Set random seed
    set_seed(42)
    
    # Get current time for experiment naming
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up logger
    experiment_name = f"{config['logging']['experiment_name']}_{current_time}"
    log_dir = os.path.join(config['logging']['log_dir'], experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    
    logger = TrainingLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        use_tensorboard=config['logging']['use_tensorboard']
    )
    
    # Save configuration
    config_save_path = os.path.join(log_dir, 'config.json')
    save_config(config, config_save_path)
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(
        config['logging']['checkpoint_dir'],
        experiment_name
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.logger.info(f"Using device: {device}")
    
    # Get dataloaders
    train_loader, val_loader, _ = get_dataloaders(config_file)
    logger.logger.info(f"Train dataset loaded with {len(train_loader.dataset)} samples")
    logger.logger.info(f"Validation dataset loaded with {len(val_loader.dataset)} samples")
    
    # Get model
    model = get_model(config)
    
    # Log depth estimator training status
    freeze_depth = config['model'].get('freeze_depth_estimator', True)
    depth_loss_weight = config['training'].get('depth_loss_weight', 0.0)
    logger.logger.info(f"Depth estimator frozen: {freeze_depth}")
    logger.logger.info(f"Depth loss weight: {depth_loss_weight}")
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config['training']['lr']),
        weight_decay=float(config['training']['weight_decay'])
    )
    
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
        val_loss, val_metrics, depth_metrics = validate(
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
            'depth_metrics': depth_metrics,
            'config': config
        }, latest_checkpoint_path)
        
        # Check for best model by validation loss and F1 score
        improved = False

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
                'depth_metrics': depth_metrics,
                'config': config
            }, best_loss_checkpoint_path)
            logger.logger.info(f"New best model (loss) saved at epoch {epoch}")
            improved = True

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
                'depth_metrics': depth_metrics,
                'config': config
            }, best_f1_checkpoint_path)
            logger.logger.info(f"New best model (F1 score) saved at epoch {epoch}")
            improved = True

        # Update early stopping counter only if neither metric improved
        if improved:
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        # Early stopping
        if early_stopping_counter >= patience:
            logger.logger.info(f"Early stopping triggered after {patience} epochs without improvement in both loss and F1 score")
            break
    
    # Close logger
    logger.close()
    
    logger.logger.info("Training completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Pipeline B: Monocular Depth Estimation + Classification')
    parser.add_argument('--config', type=str, default='src/pipelineB/config.yaml', help='Path to configuration file')
    args = parser.parse_args()
    
    main(args.config) 