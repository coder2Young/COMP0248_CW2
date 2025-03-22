import os
import sys
import json

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import time
import yaml
import random
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from src.pipelineA.model import get_model
from src.pipelineA.dataset import get_dataloaders, load_config
from src.utils.metrics import compute_metrics_from_logits, compute_classification_metrics
from src.utils.logging import TrainingLogger
from src.utils.visualization import visualize_point_cloud, plot_confusion_matrix, plot_metrics_comparison

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
            try:
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
            except Exception as e:
                print(f"Error processing batch {batch_idx} in epoch {epoch}: {e}")
                continue
    
    # Compute average loss
    epoch_loss /= len(train_loader)
    
    # Compute metrics for the entire epoch
    if all_preds and all_targets:
        # Create a proper tensor for metrics computation
        # Note: compute_metrics_from_logits expects logits, not predictions
        # Since we already have predictions, we'll compute metrics directly
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

def validate(model, dataloader, criterion, device, epoch, config):
    """
    Validate the model on the validation set.
    
    Args:
        model (torch.nn.Module): Model to validate
        dataloader (torch.utils.data.DataLoader): Validation data loader
        criterion (torch.nn.Module): Loss function
        device (torch.device): Device to use for validation
        epoch (int): Current epoch
        config (dict): Configuration dictionary
    
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
    
    # Print metrics
    print(f"Validation Loss: {val_loss:.6f}")
    print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Validation Precision: {val_metrics['precision']:.4f}")
    print(f"Validation Recall: {val_metrics['recall']:.4f}")
    print(f"Validation F1 Score: {val_metrics['f1_score']:.4f}")
    
    # Create a confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    print(f"Confusion Matrix:\n{cm}")
    
    # Visualize results periodically
    if epoch % config.get('visualization', {}).get('vis_frequency', 10) == 0:
        # Create output directory for epoch visualizations
        epoch_vis_dir = os.path.join(
            config['logging']['log_dir'],
            config['logging']['experiment_name'],
            'visualizations',
            f'epoch_{epoch}'
        )
        os.makedirs(epoch_vis_dir, exist_ok=True)
        
        # Plot confusion matrix
        cm_plot_file = os.path.join(epoch_vis_dir, 'confusion_matrix.png')
        plot_confusion_matrix(
            conf_matrix=cm,
            class_names=['No Table', 'Table'],
            title=f'Confusion Matrix - Epoch {epoch}',
            save_path=cm_plot_file
        )
        
        # Plot metrics
        metrics_plot_file = os.path.join(epoch_vis_dir, 'metrics.png')
        metrics_to_plot = {
            'Accuracy': val_metrics['accuracy'],
            'Precision': val_metrics['precision'], 
            'Recall': val_metrics['recall'],
            'F1 Score': val_metrics['f1_score'],
            'Specificity': val_metrics.get('specificity', 0)
        }
        plot_metrics_comparison(
            metrics_dict=metrics_to_plot,
            title=f"Performance Metrics - Epoch {epoch}",
            save_path=metrics_plot_file
        )
    
    return val_loss, val_metrics

def visualize_results(model, dataloader, device, config, num_samples=10):
    """
    Visualize classification results during training.
    
    Args:
        model (torch.nn.Module): Model to visualize results for
        dataloader (torch.utils.data.DataLoader): Data loader
        device (torch.device): Device to use for evaluation
        config (dict): Configuration dictionary
        num_samples (int): Number of samples to visualize
    """
    model.eval()
    
    # Check if visualization is enabled
    visualization_config = config.get('visualization', {})
    if not visualization_config.get('enabled', True):
        return
        
    # Check specific visualization types
    rgb_viz_enabled = visualization_config.get('rgb_visualization', True)
    pc_viz_enabled = visualization_config.get('point_cloud_visualization', True)
    
    if not rgb_viz_enabled and not pc_viz_enabled:
        print("Both RGB and point cloud visualizations are disabled. Skipping visualization.")
        return
    
    # Create output directory
    output_dir = os.path.join(
        config['logging']['log_dir'],
        config['logging']['experiment_name'],
        'visualizations'
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Class names for visualization
    class_names = ['No Table', 'Table']
    
    # List to store samples for visualization
    vis_samples = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            # Get the point cloud, targets, and additional metadata
            point_cloud = data['point_cloud'].to(device)
            targets = data['label'].to(device)
            
            # Get sequence and subdir information if available
            sequences = data.get('sequence', ['unknown'] * len(point_cloud))
            subdirs = data.get('subdir', ['unknown'] * len(point_cloud))
            
            # Forward pass
            outputs = model(point_cloud)
            
            # Convert outputs to predictions
            preds = torch.argmax(outputs, dim=1)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Add samples to visualization list
            for i in range(len(point_cloud)):
                if len(vis_samples) >= num_samples:
                    break
                
                vis_samples.append({
                    'point_cloud': point_cloud[i].cpu().numpy(),
                    'target': targets[i].cpu().item(),
                    'pred': preds[i].cpu().item(),
                    'confidence': probabilities[i][preds[i]].cpu().item(),
                    'sequence': sequences[i] if isinstance(sequences, list) else 'unknown',
                    'subdir': subdirs[i] if isinstance(subdirs, list) else 'unknown'
                })
            
            if len(vis_samples) >= num_samples:
                break
    
    print(f"Visualizing {len(vis_samples)} samples...")
    
    # Function to load RGB image for visualization
    def load_rgb_image(sequence, subdir, data_root='data/CW2-Dataset/data'):
        try:
            import cv2
            # Try to reconstruct the path to the RGB image
            rgb_dir = os.path.join(data_root, sequence, subdir, 'image')
            if os.path.exists(rgb_dir):
                # Just get any image from this directory for visualization
                rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')])
                if rgb_files:
                    rgb_path = os.path.join(rgb_dir, rgb_files[0])
                    rgb_image = cv2.imread(rgb_path)
                    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                    return rgb_image
        except Exception as e:
            print(f"Error loading RGB image: {e}")
        return None
    
    # Visualize samples
    for i, sample in enumerate(vis_samples):
        # 1. Visualize the point cloud if enabled
        if pc_viz_enabled:
            pc_vis_file = os.path.join(output_dir, f'sample_{i}_pointcloud_gt_{sample["target"]}_pred_{sample["pred"]}.png')
            
            from src.utils.visualization import visualize_point_cloud
            visualize_point_cloud(
                point_cloud=sample['point_cloud'],
                title=f'Ground Truth: {class_names[sample["target"]]}, '
                      f'Predicted: {class_names[sample["pred"]]} '
                      f'(Conf: {sample["confidence"]:.2f})',
                save_path=pc_vis_file
            )
            print(f"Saved point cloud visualization to {pc_vis_file}")
        
        # 2. Visualize the RGB image with classification results if enabled
        if rgb_viz_enabled:
            rgb_image = load_rgb_image(sample['sequence'], sample['subdir'], config['data']['root'])
            if rgb_image is not None:
                rgb_vis_file = os.path.join(output_dir, f'sample_{i}_rgb_gt_{sample["target"]}_pred_{sample["pred"]}.png')
                
                from src.utils.visualization import visualize_classification_results
                visualize_classification_results(
                    rgb_image=rgb_image,
                    ground_truth_label=sample['target'],
                    predicted_label=sample['pred'],
                    title=f'Sample {i} - {sample["sequence"]}/{sample["subdir"]}',
                    class_names=class_names,
                    confidence=sample['confidence'],
                    save_path=rgb_vis_file
                )
                
                print(f"Saved RGB visualization to {rgb_vis_file}")
            else:
                print(f"Could not load RGB image for sample {i} from {sample['sequence']}/{sample['subdir']}")

def main(config_file):
    """
    Main function for training Pipeline A.
    
    Args:
        config_file (str): Path to the YAML configuration file
    """
    # Load configuration
    config = load_config(config_file)
    
    # Print debug information about config types
    print("Configuration loaded:")
    print(f"Training LR type: {type(config['training']['lr'])}, value: {config['training']['lr']}")
    print(f"Weight decay type: {type(config['training']['weight_decay'])}, value: {config['training']['weight_decay']}")
    print(f"Data batch size type: {type(config['data']['batch_size'])}, value: {config['data']['batch_size']}")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    
    # Initialize logger
    logger = TrainingLogger(
        log_dir=config['logging']['log_dir'],
        experiment_name=config['logging']['experiment_name'],
        use_tensorboard=config['logging']['use_tensorboard']
    )
    
    # Log hyperparameters
    logger.log_hyperparameters(config)
    
    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config_file)
    
    # Get model
    model = get_model(config)
    model = model.to(device)
    
    # Log model summary
    logger.log_model_summary(model, input_size=(1, config['data']['num_points'], 4 if config['data']['use_height'] else 3))
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizer
    if config['training']['optimizer'].lower() == 'adam':
        print(f"Creating Adam optimizer with LR={config['training']['lr']} and weight_decay={config['training']['weight_decay']}")
        optimizer = optim.Adam(
            model.parameters(),
            lr=float(config['training']['lr']),
            weight_decay=float(config['training']['weight_decay'])
        )
    else:  # SGD
        print(f"Creating SGD optimizer with LR={config['training']['lr']} and weight_decay={config['training']['weight_decay']}")
        optimizer = optim.SGD(
            model.parameters(),
            lr=float(config['training']['lr']),
            momentum=0.9,
            weight_decay=float(config['training']['weight_decay'])
        )
    
    # Create scheduler
    if config['training']['scheduler'] == 'step':
        print(f"Creating StepLR scheduler with step_size={config['training']['lr_decay_step']} and gamma={config['training']['lr_decay']}")
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(config['training']['lr_decay_step']),
            gamma=float(config['training']['lr_decay'])
        )
    else:  # cosine
        print(f"Creating CosineAnnealingLR scheduler with T_max={config['training']['epochs']}")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config['training']['epochs'])
        )
    
    # Training loop
    best_val_loss = float('inf')
    patience = int(config['training']['early_stopping'])
    early_stopping_counter = 0
    print(f"Early stopping patience: {patience}")
    
    # Define a function to save model
    def save_model_fn(path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': config
        }, path)
    
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
            config=config
        )
        
        # Update learning rate
        scheduler.step()
        
        # Log epoch results
        logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            lr=scheduler.get_last_lr()[0],
            save_model_fn=save_model_fn
        )
        
        # Save checkpoint
        if config['logging']['save_checkpoint']:
            checkpoint_path = os.path.join(
                config['logging']['checkpoint_dir'],
                f'checkpoint_epoch_{epoch}.pth'
            )
            save_model_fn(checkpoint_path)
        
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping at epoch {epoch}, best epoch: {epoch - patience}")
                break
        
        # Visualize results periodically
        if (epoch + 1) % config.get('visualization', {}).get('vis_frequency', 10) == 0 or epoch == config['training']['epochs']:
            print(f"Visualizing results at epoch {epoch + 1}...")
            visualize_results(
                model=model,
                dataloader=val_loader,
                device=device,
                config=config,
                num_samples=config['visualization']['num_samples']
            )
    
    # Save final model
    final_model_path = os.path.join(
        config['logging']['checkpoint_dir'],
        'final_model.pth'
    )
    save_model_fn(final_model_path)
    
    # Close logger
    logger.close()
    
    print("Training completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Pipeline A: Point Cloud Classification')
    parser.add_argument('--config', type=str, default='src/pipelineA/config.yaml', help='Path to configuration file')
    args = parser.parse_args()
    
    main(args.config) 