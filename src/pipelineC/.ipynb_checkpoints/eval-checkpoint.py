import os
import sys

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import numpy as np
import argparse
import yaml
import json
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import cv2
import glob

from src.pipelineC.model import get_model
from src.pipelineC.dataset import get_dataloaders
from src.pipelineC.config import load_config, save_config
from src.utils.metrics import compute_segmentation_metrics
from src.utils.visualization import plot_confusion_matrix, visualize_point_cloud_segmentation, plot_metrics_comparison

# Set up logger
logger = logging.getLogger('evaluation')
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

def load_rgb_image(sequence, subdir, data_root='data/CW2-Dataset/data'):
    """
    Load RGB image for visualization.
    
    Args:
        sequence (str): Sequence name
        subdir (str): Subdirectory name
        data_root (str): Data root directory
    
    Returns:
        numpy.ndarray: RGB image or None if not found
    """
    try:
        # Try to reconstruct the path to the RGB image
        rgb_dir = os.path.join(data_root, sequence, subdir, 'image')
        
        if not os.path.exists(rgb_dir):
            return None
        
        # Get any image from this directory
        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')])
        if rgb_files:
            rgb_path = os.path.join(rgb_dir, rgb_files[0])
            rgb_image = cv2.imread(rgb_path)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            return rgb_image
    except Exception as e:
        logger.error(f"Error loading RGB image: {e}")
    return None

def evaluate(model, dataloader, criterion, device, output_dir, visualize=False, num_vis_samples=10):
    """
    Evaluate the model on the test set.
    
    Args:
        model (nn.Module): Model to evaluate
        dataloader (DataLoader): Test data loader
        criterion (nn.Module): Loss function
        device (torch.device): Device to use
        output_dir (str): Directory to save outputs
        visualize (bool): Whether to visualize predictions
        num_vis_samples (int): Number of samples to visualize
    
    Returns:
        tuple: (test_loss, test_metrics)
    """
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    vis_samples = []
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Progress bar for evaluation
    pbar = tqdm(dataloader, desc="Evaluating")
    
    with torch.no_grad():
        for batch_idx, data in enumerate(pbar):
            # Get data
            inputs = data['point_cloud'].to(device)
            targets = data['labels'].to(device)
            
            # if inputs.shape[1] == config['data']['num_points']:  # If second dimension is num_points, we need to transpose
            inputs = inputs.transpose(1, 2)  # (B, N, C) -> (B, C, N)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Update loss
            test_loss += loss.item()
            
            # Calculate predictions
            preds = torch.argmax(outputs, dim=1)
            
            # Update progress bar
            pbar.set_postfix({
                'test_loss': f"{loss.item():.4f}",
                'avg_loss': f"{test_loss / (batch_idx + 1):.4f}"
            })
            
            # Store predictions and targets for metrics calculation
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            # Store samples for visualization
            if visualize and len(vis_samples) < num_vis_samples:
                for i in range(min(inputs.size(0), num_vis_samples - len(vis_samples))):
                    # Get point features
                    point_features = data['point_features'][i].cpu().numpy() if 'point_features' in data else None
                    
                    # If no point features available, use simplified representation
                    if point_features is None:
                        point_features = np.zeros((inputs.size(2), 3))
                        point_features[:, 0] = np.linspace(-1, 1, inputs.size(2))
                        point_features[:, 1] = np.linspace(-1, 1, inputs.size(2))
                    
                    # Get RGB image if available
                    rgb_image = None
                    sequence = data.get('sequence', ['unknown'])[i] if isinstance(data.get('sequence', ['unknown']), list) else 'unknown'
                    subdir = data.get('subdir', ['unknown'])[i] if isinstance(data.get('subdir', ['unknown']), list) else 'unknown'
                    
                    # Try to load RGB image
                    if sequence != 'unknown' and subdir != 'unknown':
                        rgb_image = load_rgb_image(sequence, subdir)
                    
                    vis_samples.append({
                        'point_features': point_features,
                        'target_labels': targets[i].cpu().numpy(),
                        'predicted_labels': preds[i].cpu().numpy(),
                        'rgb_image': rgb_image,
                        'sequence': sequence,
                        'subdir': subdir
                    })
    
    # Compute average loss
    test_loss /= len(dataloader)
    
    # Compute metrics
    # Concatenate all predictions and targets
    flat_preds = np.concatenate([p.flatten() for p in all_preds])
    flat_targets = np.concatenate([t.flatten() for t in all_targets])
    
    test_metrics = compute_segmentation_metrics(flat_targets, flat_preds)
    
    # Log segmentation statistics
    table_points_gt = np.sum(flat_targets == 1)
    table_points_pred = np.sum(flat_preds == 1)
    bg_points_gt = np.sum(flat_targets == 0)
    bg_points_pred = np.sum(flat_preds == 0)
    total_points = len(flat_targets)
    
    # Print evaluation summary with standardized format
    print("\nTest Results:")
    print(f"  Loss:            {test_loss:.4f}")
    print(f"  Accuracy:        {test_metrics['accuracy']:.4f}")
    print(f"  Mean IoU:        {test_metrics['mean_iou']:.4f}")
    print(f"  Background IoU:  {test_metrics['iou_background']:.4f}")
    print(f"  Table IoU:       {test_metrics['iou_table']:.4f}")
    print(f"  F1 Score:        {test_metrics.get('f1_weighted', 0.0):.4f}")
    
    # Print point distribution statistics
    print("\nPoint Distribution:")
    print(f"  Ground Truth: Background: {bg_points_gt} ({bg_points_gt/total_points*100:.2f}%), "
          f"Table: {table_points_gt} ({table_points_gt/total_points*100:.2f}%)")
    print(f"  Predictions: Background: {bg_points_pred} ({bg_points_pred/total_points*100:.2f}%), "
          f"Table: {table_points_pred} ({table_points_pred/total_points*100:.2f}%)")
    
    # Save metrics to file
    metrics_file = os.path.join(output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump({
            'loss': test_loss,
            'accuracy': float(test_metrics['accuracy']),
            'mean_iou': float(test_metrics['mean_iou']),
            'iou_background': float(test_metrics['iou_background']),
            'iou_table': float(test_metrics['iou_table']),
            'f1_weighted': float(test_metrics.get('f1_weighted', 0.0)),
            'precision_weighted': float(test_metrics.get('precision_weighted', 0.0)),
            'recall_weighted': float(test_metrics.get('recall_weighted', 0.0)),
            'point_distribution': {
                'ground_truth': {
                    'background': int(bg_points_gt),
                    'table': int(table_points_gt),
                    'background_percent': float(bg_points_gt/total_points*100),
                    'table_percent': float(table_points_gt/total_points*100)
                },
                'predictions': {
                    'background': int(bg_points_pred),
                    'table': int(table_points_pred),
                    'background_percent': float(bg_points_pred/total_points*100),
                    'table_percent': float(table_points_pred/total_points*100)
                }
            }
        }, f, indent=4)
    
    print(f"Metrics saved to {metrics_file}")
    
    # Plot metrics
    metrics_plot_file = os.path.join(output_dir, 'metrics_plot.png')
    metrics_to_plot = {
        'Accuracy': test_metrics['accuracy'],
        'Mean IoU': test_metrics['mean_iou'],
        'IoU (BG)': test_metrics['iou_background'],
        'IoU (Table)': test_metrics['iou_table'],
        'F1 Score': test_metrics.get('f1_weighted', 0.0)
    }
    plot_metrics_comparison(
        metrics_dict=metrics_to_plot,
        title="Segmentation Performance Metrics",
        save_path=metrics_plot_file
    )
    
    # Visualize segmentation results
    if visualize and vis_samples:
        print(f"Visualizing {len(vis_samples)} samples...")
        
        for i, sample in enumerate(vis_samples):
            vis_file = os.path.join(vis_dir, f'sample_{i}_seq_{sample["sequence"]}.png')
            
            visualize_point_cloud_segmentation(
                point_cloud=sample['point_features'][:, :3],  # XYZ coordinates
                point_colors=sample['point_features'][:, 3:6] if sample['point_features'].shape[1] >= 6 else None,  # RGB colors if available
                target_labels=sample['target_labels'],
                pred_labels=sample['predicted_labels'],
                rgb_image=sample['rgb_image'],
                title=f"Sample {i} - {sample['sequence']}/{sample['subdir']}",
                save_path=vis_file
            )
            
            print(f"Saved visualization to {vis_file}")
    
    return test_loss, test_metrics

def main(args):
    """
    Main function for evaluating Pipeline C.
    
    Args:
        args: Command line arguments
    """
    # Load configuration
    config = load_config(args.config)
    
    # Set up file logger
    output_dir = os.path.join(config['logging']['log_dir'], 'evaluation')
    os.makedirs(output_dir, exist_ok=True)
    
    file_handler = logging.FileHandler(os.path.join(output_dir, 'evaluation.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log configuration
    logger.info(f"Evaluation configuration: {config}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Get dataloaders
    _, _, test_loader = get_dataloaders(args.config)
    logger.info(f"Test dataset loaded with {len(test_loader.dataset)} samples")
    
    # Get model
    model = get_model(config)
    model = model.to(device)
    
    # Load checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Use the best model by default
        # First, look for experiment directories
        experiment_dirs = glob.glob(os.path.join(config['logging']['log_dir'], f"{config['logging']['experiment_name']}_*"))
        if experiment_dirs:
            # Use the most recent experiment
            latest_dir = max(experiment_dirs, key=os.path.getctime)
            experiment_name = os.path.basename(latest_dir)
            
            # Look for the checkpoint directory
            checkpoint_dir = os.path.join(config['logging']['checkpoint_dir'], experiment_name)
            if os.path.exists(checkpoint_dir):
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            else:
                checkpoint_path = None
        else:
            checkpoint_path = None
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        # Try looking in all checkpoint directories
        checkpoint_dirs = glob.glob(os.path.join(config['logging']['checkpoint_dir'], "*"))
        if checkpoint_dirs:
            # Use the most recent checkpoint directory
            latest_dir = max(checkpoint_dirs, key=os.path.getctime)
            checkpoint_path = os.path.join(latest_dir, 'best_model.pth')
            if not os.path.exists(checkpoint_path):
                checkpoint_path = os.path.join(latest_dir, 'latest_model.pth')
        
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError("Could not find a checkpoint to load. Please specify one with --checkpoint.")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    # Get loss weights from the checkpoint if they exist
    if 'config' in checkpoint and 'loss' in checkpoint['config'] and 'class_weights' in checkpoint['config']['loss']:
        class_weights = torch.tensor(checkpoint['config']['loss']['class_weights'], dtype=torch.float32).to(device)
        logger.info(f"Using class weights from checkpoint: {class_weights.cpu().numpy()}")
    else:
        class_weights = None
    
    # Define loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Evaluate model
    test_loss, test_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        output_dir=output_dir,
        visualize=args.visualize,
        num_vis_samples=args.num_vis_samples
    )
    
    logger.info("Evaluation completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Pipeline C: Point Cloud Segmentation')
    parser.add_argument('--config', type=str, default='src/pipelineC/config.yaml', help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    parser.add_argument('--num_vis_samples', type=int, default=10, help='Number of samples to visualize')
    args = parser.parse_args()
    
    main(args) 