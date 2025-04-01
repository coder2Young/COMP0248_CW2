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
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import cv2
import glob

from src.pipelineB.model import get_model
from src.pipelineB.dataset import get_dataloaders
from src.pipelineB.config import load_config, save_config
from src.utils.metrics import compute_classification_metrics, compute_depth_metrics
from src.utils.visualization import plot_confusion_matrix, visualize_classification_results, plot_metrics_comparison

# Set up logger
logger = logging.getLogger('evaluation')
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

def load_rgb_image(sequence, subdir, file_path, data_root='data/CW2-Dataset/data'):
    """
    Load RGB image for visualization.
    
    Args:
        sequence (str): Sequence name
        subdir (str): Subdirectory name
        file_path (str): Image file name
        data_root (str): Data root directory
    
    Returns:
        numpy.ndarray: RGB image or None if not found
    """
    try:
        # Try to reconstruct the path to the RGB image
        rgb_dir = os.path.join(data_root, sequence, subdir, 'image')
        rgb_path = os.path.join(rgb_dir, file_path)
        
        # If specific file path is not found, just get any image from this directory
        if not os.path.exists(rgb_path):
            rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')])
            if rgb_files:
                rgb_path = os.path.join(rgb_dir, rgb_files[0])
        
        if os.path.exists(rgb_path):
            rgb_image = cv2.imread(rgb_path)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            return rgb_image
    except Exception as e:
        logger.error(f"Error loading RGB image: {e}")
    return None

def visualize_depth_classification(rgb_image, depth_map, ground_truth_label=None, predicted_label=None, 
                                 confidence=None, class_names=None, title='Depth Classification Results', 
                                 save_path=None, figsize=(15, 5)):
    """
    Visualize depth classification results.
    
    Args:
        rgb_image (numpy.ndarray): RGB image
        depth_map (numpy.ndarray): Depth map
        ground_truth_label (int, optional): Ground truth label
        predicted_label (int, optional): Predicted label
        confidence (float, optional): Confidence score for the prediction
        class_names (list, optional): List of class names
        title (str, optional): Title for the plot
        save_path (str, optional): Path to save the plot
        figsize (tuple, optional): Figure size
    """
    if class_names is None:
        class_names = ['No Table', 'Table']
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot RGB image
    axes[0].imshow(rgb_image)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')
    
    # Plot depth map
    if depth_map is not None:
        # Normalize depth map for visualization if needed
        if depth_map.max() > 1.0:
            depth_map = depth_map / depth_map.max()
        axes[1].imshow(depth_map, cmap='plasma')
        axes[1].set_title('Estimated Depth Map')
        axes[1].axis('off')
    else:
        axes[1].set_title('Depth Map Not Available')
        axes[1].axis('off')
    
    # Plot classification results
    axes[2].imshow(np.ones((10, 10, 3)))  # Placeholder image
    axes[2].axis('off')
    
    # Add text for ground truth and prediction
    result_text = ""
    
    if ground_truth_label is not None:
        gt_text = f"Ground Truth: {class_names[ground_truth_label]}"
        result_text += gt_text + "\n"
    
    if predicted_label is not None:
        pred_text = f"Prediction: {class_names[predicted_label]}"
        if confidence is not None:
            pred_text += f" ({confidence:.2f})"
        result_text += pred_text + "\n"
        
        # Add color-coded correctness indicator
        if ground_truth_label is not None:
            if ground_truth_label == predicted_label:
                result_text += "Correct ✓"
                axes[2].text(0.5, 0.5, result_text, ha='center', va='center', 
                           fontsize=12, color='green', transform=axes[2].transAxes)
            else:
                result_text += "Incorrect ✗"
                axes[2].text(0.5, 0.5, result_text, ha='center', va='center', 
                           fontsize=12, color='red', transform=axes[2].transAxes)
        else:
            axes[2].text(0.5, 0.5, result_text, ha='center', va='center', 
                       fontsize=12, transform=axes[2].transAxes)
    
    # Add overall title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_depth_classification_with_gt(rgb_image, pred_depth, gt_depth=None, 
                                         ground_truth_label=None, predicted_label=None, 
                                         confidence=None, class_names=None, title='Depth Classification Results', 
                                         save_path=None, figsize=(20, 5)):
    """
    Visualize depth classification results with ground truth depth comparison.
    
    Args:
        rgb_image (numpy.ndarray): RGB image
        pred_depth (numpy.ndarray): Predicted depth map
        gt_depth (numpy.ndarray, optional): Ground truth depth map
        ground_truth_label (int, optional): Ground truth label
        predicted_label (int, optional): Predicted label
        confidence (float, optional): Confidence score for the prediction
        class_names (list, optional): List of class names
        title (str, optional): Title for the plot
        save_path (str, optional): Path to save the plot
        figsize (tuple, optional): Figure size
    """
    if class_names is None:
        class_names = ['No Table', 'Table']
    
    # Determine if we have GT depth (4 columns) or not (3 columns)
    n_cols = 4 if gt_depth is not None else 3
    
    # Create figure
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    
    # Plot RGB image
    axes[0].imshow(rgb_image)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')
    
    # Plot predicted depth map
    if pred_depth is not None:
        # Normalize depth map for visualization if needed
        if pred_depth.max() > 1.0:
            pred_depth = pred_depth / pred_depth.max()
        axes[1].imshow(pred_depth, cmap='plasma')
        axes[1].set_title('Predicted Depth Map')
        axes[1].axis('off')
    else:
        axes[1].set_title('Depth Map Not Available')
        axes[1].axis('off')
    
    # Plot ground truth depth map if available
    if gt_depth is not None:
        # Normalize depth map for visualization if needed
        if np.max(gt_depth) > 0:
            gt_depth = gt_depth / np.max(gt_depth)
        axes[2].imshow(gt_depth, cmap='plasma')
        axes[2].set_title('Ground Truth Depth Map')
        axes[2].axis('off')
        
        # Classification results go in the 4th column
        result_idx = 3
    else:
        # Classification results go in the 3rd column
        result_idx = 2
    
    # Plot classification results
    axes[result_idx].imshow(np.ones((10, 10, 3)))  # Placeholder image
    axes[result_idx].axis('off')
    
    # Add text for ground truth and prediction
    result_text = ""
    
    if ground_truth_label is not None:
        gt_text = f"Ground Truth: {class_names[ground_truth_label]}"
        result_text += gt_text + "\n"
    
    if predicted_label is not None:
        pred_text = f"Prediction: {class_names[predicted_label]}"
        if confidence is not None:
            pred_text += f" ({confidence:.2f})"
        result_text += pred_text + "\n"
        
        # Add color-coded correctness indicator
        if ground_truth_label is not None:
            if ground_truth_label == predicted_label:
                result_text += "Correct ✓"
                axes[result_idx].text(0.5, 0.5, result_text, ha='center', va='center', 
                           fontsize=12, color='green', transform=axes[result_idx].transAxes)
            else:
                result_text += "Incorrect ✗"
                axes[result_idx].text(0.5, 0.5, result_text, ha='center', va='center', 
                           fontsize=12, color='red', transform=axes[result_idx].transAxes)
        else:
            axes[result_idx].text(0.5, 0.5, result_text, ha='center', va='center', 
                       fontsize=12, transform=axes[result_idx].transAxes)
    
    # Add overall title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def evaluate(model, dataloader, criterion, device, output_dir, visualize=False, num_vis_samples=20):
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
        tuple: (test_loss, test_metrics, depth_metrics)
    """
    model.eval()
    test_loss = 0.0
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
    
    # Sample info for debugging
    sample_info = []
    
    # Create directories for outputs
    os.makedirs(output_dir, exist_ok=True)
    if visualize:
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
    
    # Sample data for visualization
    vis_data = []
    
    logger.info("Starting evaluation...")
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dataloader, desc="Testing")):
            # Get data
            inputs = data['rgb_image'].to(device)
            targets = data['label'].to(device)
            
            # Get ground truth depth if available
            gt_depth = data.get('gt_depth')
            if gt_depth is not None:
                gt_depth = gt_depth.to(device)
            
            # Try to access sequence and subdir information if available
            sequences = data.get('sequence', ['unknown'] * len(inputs))
            subdirs = data.get('subdir', ['unknown'] * len(inputs))
            file_paths = data.get('file_path', ['unknown'] * len(inputs))
            
            # Forward pass with depth maps
            outputs, pred_depth = model(inputs, return_depth=True)
            loss = criterion(outputs, targets)
            
            # Update metrics
            test_loss += loss.item()
            
            # Convert outputs to predictions
            preds = torch.argmax(outputs, dim=1)
            probabilities = torch.softmax(outputs, dim=1)
            
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
                    metrics = compute_depth_metrics(sample_pred_depth, sample_gt_depth)
                    
                    # Skip samples with invalid metrics
                    if any(np.isnan(list(metrics.values()))):
                        continue
                    
                    # Accumulate metrics
                    for k, v in metrics.items():
                        depth_errors[k].append(v)
                    
                    valid_depth_samples += 1
            
            # Store detailed sample info for debugging
            for i in range(len(inputs)):
                sample_info.append({
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'sequence': sequences[i] if isinstance(sequences, list) else 'unknown',
                    'subdir': subdirs[i] if isinstance(subdirs, list) else 'unknown',
                    'file_path': file_paths[i] if isinstance(file_paths, list) else 'unknown',
                    'target': targets[i].cpu().item(),
                    'prediction': preds[i].cpu().item(),
                    'confidence': probabilities[i][preds[i]].cpu().item()
                })
            
            # Store some samples for visualization
            if visualize and len(vis_data) < num_vis_samples:
                for i in range(min(len(inputs), num_vis_samples - len(vis_data))):
                    # Store sample for visualization
                    sample_gt_depth = gt_depth[i].cpu() if gt_depth is not None else None
                    
                    vis_data.append({
                        'rgb_tensor': inputs[i].cpu(),
                        'pred_depth': pred_depth[i].cpu(),
                        'gt_depth': sample_gt_depth,
                        'target': targets[i].cpu().item(),
                        'pred': preds[i].cpu().item(),
                        'confidence': probabilities[i][preds[i]].cpu().item(),
                        'sequence': sequences[i] if isinstance(sequences, list) else 'unknown',
                        'subdir': subdirs[i] if isinstance(subdirs, list) else 'unknown',
                        'file_path': file_paths[i] if isinstance(file_paths, list) else 'unknown',
                        'batch_idx': batch_idx,
                        'sample_idx': i
                    })
    
    # Compute average loss
    test_loss /= len(dataloader)
    
    # Compute classification metrics
    test_metrics = compute_classification_metrics(
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
    
    # Log depth evaluation results
    if depth_metrics:
        logger.info("===== Depth Estimation Metrics =====")
        logger.info(f"Valid depth samples: {valid_depth_samples}")
        logger.info(f"RMSE:  {depth_metrics.get('rmse', float('nan')):.4f} meters")
        logger.info(f"MAE:   {depth_metrics.get('mae', float('nan')):.4f} meters")
        logger.info(f"REL:   {depth_metrics.get('rel', float('nan')):.4f}")
        logger.info(f"δ1:    {depth_metrics.get('a1', float('nan')):.4f} (% under 1.25)")
        logger.info(f"δ2:    {depth_metrics.get('a2', float('nan')):.4f} (% under 1.25²)")
        logger.info(f"δ3:    {depth_metrics.get('a3', float('nan')):.4f} (% under 1.25³)")
    else:
        logger.info("No valid depth samples for evaluation.")
    
    # Log label distribution
    target_counts = np.bincount(np.array(all_targets), minlength=2)
    pred_counts = np.bincount(np.array(all_preds), minlength=2)
    
    logger.info("===== Label Distribution Statistics =====")
    logger.info(f"Ground Truth: No Table (0): {target_counts[0]}, Table (1): {target_counts[1]}")
    logger.info(f"Predictions: No Table (0): {pred_counts[0]}, Table (1): {pred_counts[1]}")
    
    # Save detailed sample info to CSV
    sample_info_file = os.path.join(output_dir, 'sample_predictions.csv')
    with open(sample_info_file, 'w') as f:
        f.write("batch_idx,sample_idx,sequence,subdir,file_path,target,prediction,confidence\n")
        for sample in sample_info:
            f.write(f"{sample['batch_idx']},{sample['sample_idx']},{sample['sequence']},"
                    f"{sample['subdir']},{sample['file_path']},{sample['target']},"
                    f"{sample['prediction']},{sample['confidence']:.4f}\n")
    
    # Print first 20 predictions for manual verification
    logger.info("===== First 20 Sample Predictions =====")
    logger.info("Idx\tSequence\tSubdir\tGround Truth\tPrediction\tConfidence")
    for i, sample in enumerate(sample_info[:20]):
        logger.info(f"{i}\t{sample['sequence']}\t{sample['subdir']}\t{sample['target']}\t{sample['prediction']}\t{sample['confidence']:.4f}")
    
    # Print metrics
    logger.info("===== Classification Metrics =====")
    logger.info(f"Test Loss:       {test_loss:.6f}")
    logger.info(f"Test Accuracy:   {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Precision:  {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall:     {test_metrics['recall']:.4f}")
    logger.info(f"Test F1 Score:   {test_metrics['f1_score']:.4f}")
    logger.info(f"Test Specificity: {test_metrics['specificity']:.4f}")
    
    # Print confusion matrix for better visualization
    cm = confusion_matrix(all_targets, all_preds)
    logger.info("===== Confusion Matrix =====")
    logger.info("GT\\Pred\tNo Table(0)\tTable(1)")
    logger.info(f"No Table(0)\t{cm[0][0]}\t\t{cm[0][1]}")
    logger.info(f"Table(1)\t{cm[1][0]}\t\t{cm[1][1]}")
    
    # Convert NumPy types to native Python types for JSON serialization
    serializable_metrics = {}
    for key, value in test_metrics.items():
        if isinstance(value, np.integer):
            serializable_metrics[key] = int(value)
        elif isinstance(value, np.floating):
            serializable_metrics[key] = float(value)
        elif isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        else:
            serializable_metrics[key] = value
    
    # Add depth metrics to serializable metrics
    serializable_metrics['depth'] = {k: float(v) if not np.isnan(v) else None for k, v in depth_metrics.items()}
    
    # Save metrics to JSON file
    metrics_file = os.path.join(output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump({
            'loss': float(test_loss),
            **serializable_metrics
        }, f, indent=4)
    
    # Update visualization function to include ground truth depth
    if visualize:
        logger.info(f"Visualizing {len(vis_data)} samples with RGB images, predicted depths, and ground truth depths...")
        for i, sample in enumerate(vis_data):
            # Get original RGB image
            rgb_image = load_rgb_image(
                sequence=sample['sequence'], 
                subdir=sample['subdir'],
                file_path=sample['file_path'],
                data_root=dataloader.dataset.dataset.data_root 
                    if hasattr(dataloader.dataset, 'dataset') else 'data/CW2-Dataset/data'
            )
            
            if rgb_image is not None:
                # Set up the visualization path
                vis_file = os.path.join(vis_dir, f'sample_{i}_gt_{sample["target"]}_pred_{sample["pred"]}.png')
                
                # Visualize with depth comparison
                visualize_depth_classification_with_gt(
                    rgb_image=rgb_image,
                    pred_depth=sample['pred_depth'].numpy(),
                    gt_depth=sample['gt_depth'].numpy() if sample['gt_depth'] is not None else None,
                    ground_truth_label=sample['target'],
                    predicted_label=sample['pred'],
                    confidence=sample['confidence'],
                    class_names=['No Table', 'Table'],
                    title=f'Sample {i} - {sample["sequence"]}/{sample["subdir"]}',
                    save_path=vis_file
                )
                
                logger.info(f"Saved visualization to {vis_file}")
    
    return test_loss, test_metrics, depth_metrics

def main(args):
    """
    Main function for evaluating Pipeline B.
    
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
        checkpoint_path = os.path.join(
            config['logging']['checkpoint_dir'],
            config['logging']['experiment_name'],
            'best_model.pth'
        )
    
    if not os.path.exists(checkpoint_path):
        # Try to find the checkpoint in a timestamped directory
        experiment_dirs = glob.glob(os.path.join(config['logging']['log_dir'], f"{config['logging']['experiment_name']}_*"))
        if experiment_dirs:
            # Use the most recent experiment
            latest_dir = max(experiment_dirs, key=os.path.getctime)
            potential_checkpoint = os.path.join(latest_dir, 'best_model.pth')
            if os.path.exists(potential_checkpoint):
                checkpoint_path = potential_checkpoint
            else:
                # Try the checkpoint directory
                checkpoint_dirs = glob.glob(os.path.join(config['logging']['checkpoint_dir'], f"{config['logging']['experiment_name']}_*"))
                if checkpoint_dirs:
                    latest_checkpoint_dir = max(checkpoint_dirs, key=os.path.getctime)
                    potential_checkpoint = os.path.join(latest_checkpoint_dir, 'best_model.pth')
                    if os.path.exists(potential_checkpoint):
                        checkpoint_path = potential_checkpoint
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate model
    test_loss, test_metrics, depth_metrics = evaluate(
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
    parser = argparse.ArgumentParser(description='Evaluate Pipeline B: Monocular Depth Estimation + Classification')
    parser.add_argument('--config', type=str, default='src/pipelineB/config.yaml', help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    parser.add_argument('--num_vis_samples', type=int, default=20, help='Number of samples to visualize')
    args = parser.parse_args()
    
    main(args) 