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
        model (torch.nn.Module): Model to evaluate
        dataloader (torch.utils.data.DataLoader): Test data loader
        criterion (torch.nn.Module): Loss function
        device (torch.device): Device to use for evaluation
        output_dir (str): Directory to save results
        visualize (bool): Whether to visualize results
        num_vis_samples (int): Number of samples to visualize
    
    Returns:
        tuple: (test_loss, test_metrics)
    """
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    
    # For visualization
    vis_samples = []
    
    # Create directories for outputs
    os.makedirs(output_dir, exist_ok=True)
    if visualize:
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
    
    logger.info("Starting evaluation...")
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dataloader, desc="Testing")):
            # Get data
            inputs = data['point_features'].permute(0, 2, 1).to(device)  # (B, C, N)
            targets = data['point_labels'].to(device)  # (B, N)
            
            # Forward pass
            outputs = model(inputs)  # (B, num_classes, N)
            loss = criterion(outputs, targets)
            
            # Convert outputs to predictions
            preds = torch.argmax(outputs, dim=1)  # (B, N)
            
            # Check if model is making varied predictions
            num_unique_preds = torch.unique(preds).size(0)
            if num_unique_preds == 1:
                logger.info(f"Evaluation: Model predicting only one class: {preds[0, 0].item()}")
            
            # Update metrics
            test_loss += loss.item()
            
            # Store predictions and targets for metrics computation
            for i in range(targets.size(0)):
                all_preds.append(preds[i].cpu().numpy())
                all_targets.append(targets[i].cpu().numpy())
            
            # Store samples for visualization
            if visualize and len(vis_samples) < num_vis_samples:
                for i in range(min(inputs.size(0), num_vis_samples - len(vis_samples))):
                    # Get the original point cloud
                    point_features = data['point_features'][i].cpu().numpy()
                    
                    # Get sequence and subdir info
                    sequence = None
                    subdir = None
                    
                    if hasattr(dataloader.dataset, 'dataset'):
                        # For data.Subset
                        if hasattr(dataloader.dataset.dataset, 'data_pairs'):
                            if batch_idx * dataloader.batch_size + i < len(dataloader.dataset.dataset.data_pairs):
                                idx = dataloader.dataset.indices[batch_idx * dataloader.batch_size + i]
                                sequence = dataloader.dataset.dataset.data_pairs[idx].get('sequence', None)
                                subdir = dataloader.dataset.dataset.data_pairs[idx].get('subdir', None)
                    elif hasattr(dataloader.dataset, 'data_pairs'):
                        # For direct dataset
                        if batch_idx * dataloader.batch_size + i < len(dataloader.dataset.data_pairs):
                            sequence = dataloader.dataset.data_pairs[batch_idx * dataloader.batch_size + i].get('sequence', None)
                            subdir = dataloader.dataset.data_pairs[batch_idx * dataloader.batch_size + i].get('subdir', None)
                    
                    # Get RGB image if sequence and subdir are available
                    rgb_image = None
                    if sequence and subdir:
                        rgb_image = load_rgb_image(sequence, subdir, dataloader.dataset.dataset.data_root 
                            if hasattr(dataloader.dataset, 'dataset') else dataloader.dataset.data_root)
                    
                    vis_samples.append({
                        'point_features': point_features,
                        'target_labels': targets[i].cpu().numpy(),
                        'predicted_labels': preds[i].cpu().numpy(),
                        'rgb_image': rgb_image,
                        'sequence': sequence if sequence else 'unknown',
                        'subdir': subdir if subdir else 'unknown'
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
    
    logger.info("===== Point Cloud Segmentation Statistics =====")
    logger.info(f"Total points evaluated: {total_points}")
    logger.info(f"Ground Truth: Background: {bg_points_gt} ({bg_points_gt/total_points*100:.2f}%), "
              f"Table: {table_points_gt} ({table_points_gt/total_points*100:.2f}%)")
    logger.info(f"Predictions: Background: {bg_points_pred} ({bg_points_pred/total_points*100:.2f}%), "
              f"Table: {table_points_pred} ({table_points_pred/total_points*100:.2f}%)")
    
    # Log metrics in a vertical format for better readability
    logger.info("===== Evaluation Metrics =====")
    logger.info(f"Test Loss:          {test_loss:.6f}")
    logger.info(f"Accuracy:           {test_metrics['accuracy']:.4f}")
    logger.info(f"IoU (Background):   {test_metrics.get('iou_0', 0.0):.4f}")
    logger.info(f"IoU (Table):        {test_metrics.get('iou_1', 0.0):.4f}")
    logger.info(f"Mean IoU:           {test_metrics['mean_iou']:.4f}")
    logger.info(f"Precision:          {test_metrics.get('precision_weighted', test_metrics.get('precision_macro', 0.0)):.4f}")
    logger.info(f"Recall:             {test_metrics.get('recall_weighted', test_metrics.get('recall_macro', 0.0)):.4f}")
    logger.info(f"F1 Score:           {test_metrics.get('f1_weighted', test_metrics.get('f1_macro', 0.0)):.4f}")
    
    # Print confusion matrix
    cm = confusion_matrix(flat_targets, flat_preds, labels=[0, 1])
    logger.info("===== Confusion Matrix =====")
    logger.info("GT\\Pred\tBackground(0)\tTable(1)")
    logger.info(f"BG(0)\t{cm[0][0]}\t\t{cm[0][1]}")
    logger.info(f"Table(1)\t{cm[1][0]}\t\t{cm[1][1]}")
    
    # Save metrics to JSON file
    metrics_file = os.path.join(output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump({
            'loss': float(test_loss),
            'accuracy': float(test_metrics['accuracy']),
            'iou_background': float(test_metrics.get('iou_0', 0.0)),
            'iou_table': float(test_metrics.get('iou_1', 0.0)),
            'mean_iou': float(test_metrics['mean_iou']),
            'precision': float(test_metrics.get('precision_weighted', test_metrics.get('precision_macro', 0.0))),
            'recall': float(test_metrics.get('recall_weighted', test_metrics.get('recall_macro', 0.0))),
            'f1_score': float(test_metrics.get('f1_weighted', test_metrics.get('f1_macro', 0.0))),
            'confusion_matrix': cm.tolist()
        }, f, indent=4)
    
    # Plot confusion matrix
    cm_plot_file = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(
        conf_matrix=cm,
        class_names=['Background', 'Table'],
        title='Confusion Matrix',
        save_path=cm_plot_file
    )
    
    # Plot metrics comparison
    metrics_plot_file = os.path.join(output_dir, 'metrics_comparison.png')
    metrics_to_plot = {
        'Accuracy': test_metrics['accuracy'],
        'Mean IoU': test_metrics['mean_iou'],
        'IoU (BG)': test_metrics.get('iou_0', 0.0),
        'IoU (Table)': test_metrics.get('iou_1', 0.0),
        'Precision': test_metrics.get('precision_weighted', test_metrics.get('precision_macro', 0.0)), 
        'Recall': test_metrics.get('recall_weighted', test_metrics.get('recall_macro', 0.0)),
        'F1 Score': test_metrics.get('f1_weighted', test_metrics.get('f1_macro', 0.0))
    }
    plot_metrics_comparison(
        metrics_dict=metrics_to_plot,
        title="Segmentation Performance Metrics",
        save_path=metrics_plot_file
    )
    
    # Visualize segmentation results
    if visualize and vis_samples:
        logger.info(f"Visualizing {len(vis_samples)} samples...")
        
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
            
            logger.info(f"Saved visualization to {vis_file}")
    
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