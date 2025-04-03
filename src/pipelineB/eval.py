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
from src.data_utils.realsense_dataset import get_realsense_dataloader

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

def evaluate(model, device, config_file, output_dir=None, visualize=False, num_vis_samples=10):
    """
    Evaluate the model on the test set.
    
    Args:
        model (nn.Module): Model to evaluate
        device (torch.device): Device to use
        config_file (str): Path to the configuration file
        output_dir (str, optional): Directory to save results
        visualize (bool): Whether to visualize results
        num_vis_samples (int): Number of samples to visualize
    
    Returns:
        dict: Dictionary of metrics
    """
    # Load configuration
    config = load_config(config_file)
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(config['logging']['log_dir'], config['logging']['experiment_name'])
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get test dataloader
    # Use eval_batch_size for testing
    test_dataloader = get_dataloader(config, 'test', transform=None, batch_size=config['data'].get('eval_batch_size', config['data'].get('batch_size', 4)))
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize metrics
    all_labels = []
    all_preds = []
    
    # Initialize visualization
    if visualize:
        vis_dir = os.path.join(output_dir, 'visualization')
        os.makedirs(vis_dir, exist_ok=True)
        vis_count = 0
    
    # Evaluate model
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_dataloader)):
            # Get inputs and labels
            inputs = data['rgb_image'].to(device)
            labels = data['binary_label'].to(device)
            
            # Forward pass
            outputs, depth_maps = model(inputs, return_depth=True)
            
            # Get predictions
            preds = torch.argmax(outputs, dim=1)
            
            # Append to lists for metrics computation
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            
            # Visualize if needed
            if visualize and vis_count < num_vis_samples:
                for i in range(min(inputs.size(0), num_vis_samples - vis_count)):
                    # Get image, depth map, and prediction
                    image = inputs[i].cpu().numpy().transpose(1, 2, 0)
                    depth_map = depth_maps[i].cpu().numpy()
                    pred = preds[i].item()
                    label = labels[i].item()
                    
                    # Normalize image for visualization
                    image = (image * 255).astype(np.uint8)
                    
                    # Normalize depth map for visualization
                    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
                    depth_map = (depth_map * 255).astype(np.uint8)
                    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
                    
                    # Create visualization
                    if pred == label:
                        result_text = "Correct"
                        color = (0, 255, 0)  # Green for correct
                    else:
                        result_text = "Wrong"
                        color = (255, 0, 0)  # Red for wrong
                    
                    # Add text to image
                    image = cv2.putText(
                        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                        f"Pred: {pred}, GT: {label} ({result_text})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )
                    
                    # Save visualization
                    vis_path = os.path.join(vis_dir, f"sample_{batch_idx}_{i}.png")
                    cv2.imwrite(vis_path, image)
                    
                    # Save depth map
                    depth_path = os.path.join(vis_dir, f"depth_{batch_idx}_{i}.png")
                    cv2.imwrite(depth_path, depth_map)
                    
                    vis_count += 1
    
    # Concatenate all labels and predictions
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    
    # Compute metrics
    metrics = compute_classification_metrics(all_labels, all_preds)
    
    # Print metrics
    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

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
        device=device,
        config_file=args.config,
        output_dir=output_dir,
        visualize=args.visualize,
        num_vis_samples=args.num_vis_samples
    )
    
    # Evaluate on Sun3D dataset
    logger.info("===== Evaluating on Sun3D dataset =====")
    sun3d_loss, sun3d_metrics, sun3d_depth_metrics = evaluate(
        model=model,
        device=device,
        config_file=args.config,
        output_dir=os.path.join(output_dir, 'sun3d'),
        visualize=args.visualize,
        num_vis_samples=args.num_vis_samples
    )
    
    # Evaluate on RealSense dataset if requested
    if args.eval_realsense:
        logger.info("===== Evaluating on RealSense dataset =====")
        
        # Get RealSense dataloader
        realsense_loader = get_realsense_dataloader(config, pipeline='pipelineB')
        logger.info(f"RealSense dataset loaded with {len(realsense_loader.dataset)} samples")
        
        # Evaluate on RealSense dataset
        realsense_loss, realsense_metrics, realsense_depth_metrics = evaluate(
            model=model,
            device=device,
            config_file=args.config,
            output_dir=os.path.join(output_dir, 'realsense'),
            visualize=args.visualize,
            num_vis_samples=args.num_vis_samples
        )
        
        # Print comparison between datasets for classification
        logger.info("===== Classification Metrics Comparison =====")
        logger.info(f"Metric\t\tSun3D\t\tRealSense")
        logger.info(f"Loss\t\t{sun3d_loss:.4f}\t\t{realsense_loss:.4f}")
        logger.info(f"Accuracy\t{sun3d_metrics['accuracy']:.4f}\t\t{realsense_metrics['accuracy']:.4f}")
        logger.info(f"Precision\t{sun3d_metrics['precision']:.4f}\t\t{realsense_metrics['precision']:.4f}")
        logger.info(f"Recall\t\t{sun3d_metrics['recall']:.4f}\t\t{realsense_metrics['recall']:.4f}")
        logger.info(f"F1 Score\t{sun3d_metrics['f1_score']:.4f}\t\t{realsense_metrics['f1_score']:.4f}")
        logger.info(f"Specificity\t{sun3d_metrics['specificity']:.4f}\t\t{realsense_metrics['specificity']:.4f}")
        
        # Print comparison for depth metrics if available
        if sun3d_depth_metrics and realsense_depth_metrics:
            logger.info("===== Depth Metrics Comparison =====")
            logger.info(f"Metric\t\tSun3D\t\tRealSense")
            for metric in sun3d_depth_metrics:
                if metric in realsense_depth_metrics:
                    logger.info(f"{metric}\t\t{sun3d_depth_metrics[metric]:.4f}\t\t{realsense_depth_metrics[metric]:.4f}")
        
        # Save comparison results
        comparison_file = os.path.join(output_dir, 'dataset_comparison.json')
        comparison_data = {
            'sun3d': {
                'loss': float(sun3d_loss),
                'classification': {k: float(v) for k, v in sun3d_metrics.items()},
                'depth': {k: float(v) for k, v in sun3d_depth_metrics.items()} if sun3d_depth_metrics else {}
            },
            'realsense': {
                'loss': float(realsense_loss),
                'classification': {k: float(v) for k, v in realsense_metrics.items()},
                'depth': {k: float(v) for k, v in realsense_depth_metrics.items()} if realsense_depth_metrics else {}
            }
        }
        
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=4)
    
    logger.info("Evaluation completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Pipeline B: Monocular Depth Estimation + Classification')
    parser.add_argument('--config', type=str, default='src/pipelineB/config.yaml', help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    parser.add_argument('--num_vis_samples', type=int, default=20, help='Number of samples to visualize')
    parser.add_argument('--eval_realsense', action='store_true', help='Evaluate on RealSense dataset')
    args = parser.parse_args()
    
    main(args) 