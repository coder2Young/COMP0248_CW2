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

from src.pipelineA.model import get_model
from src.pipelineA.dataset import get_dataloaders
from src.pipelineA.config import load_config, save_config
from src.utils.metrics import compute_classification_metrics
from src.utils.visualization import plot_confusion_matrix, visualize_classification_results, plot_metrics_comparison
from src.data_utils.realsense_dataset import get_realsense_dataloader

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
        if os.path.exists(rgb_dir):
            # Just get any image from this directory for visualization
            rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')])
            if rgb_files:
                rgb_path = os.path.join(rgb_dir, rgb_files[0])
                rgb_image = cv2.imread(rgb_path)
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                return rgb_image
    except Exception as e:
        logger.error(f"Error loading RGB image: {e}")
    return None

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
        tuple: (test_loss, test_metrics)
    """
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    
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
            inputs = data['point_cloud'].to(device)
            targets = data['label'].to(device)
            
            # Try to access sequence and subdir information if available
            sequences = data.get('sequence', ['unknown'] * len(inputs))
            subdirs = data.get('subdir', ['unknown'] * len(inputs))
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Update metrics
            test_loss += loss.item()
            
            # Convert outputs to predictions
            preds = torch.argmax(outputs, dim=1)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Store predictions and targets for metrics computation
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Store detailed sample info for debugging
            for i in range(len(inputs)):
                sample_info.append({
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'sequence': sequences[i] if isinstance(sequences, list) else 'unknown',
                    'subdir': subdirs[i] if isinstance(subdirs, list) else 'unknown',
                    'target': targets[i].cpu().item(),
                    'prediction': preds[i].cpu().item(),
                    'confidence': probabilities[i][preds[i]].cpu().item()
                })
            
            # Store some samples for visualization
            if visualize and len(vis_data) < num_vis_samples:
                for i in range(min(len(inputs), num_vis_samples - len(vis_data))):
                    vis_data.append({
                        'point_cloud': inputs[i].cpu().numpy(),
                        'target': targets[i].cpu().item(),
                        'pred': preds[i].cpu().item(),
                        'confidence': probabilities[i][preds[i]].cpu().item(),
                        'sequence': sequences[i] if isinstance(sequences, list) else 'unknown',
                        'subdir': subdirs[i] if isinstance(subdirs, list) else 'unknown',
                        'batch_idx': batch_idx,
                        'sample_idx': i
                    })
    
    # Compute average loss
    test_loss /= len(dataloader)
    
    # Compute metrics
    test_metrics = compute_classification_metrics(
        np.array(all_targets),
        np.array(all_preds)
    )
    
    # Log label distribution
    target_counts = np.bincount(np.array(all_targets), minlength=2)
    pred_counts = np.bincount(np.array(all_preds), minlength=2)
    
    logger.info("===== Label Distribution Statistics =====")
    logger.info(f"Ground Truth: No Table (0): {target_counts[0]}, Table (1): {target_counts[1]}")
    logger.info(f"Predictions: No Table (0): {pred_counts[0]}, Table (1): {pred_counts[1]}")
    
    # Save detailed sample info to CSV
    sample_info_file = os.path.join(output_dir, 'sample_predictions.csv')
    with open(sample_info_file, 'w') as f:
        f.write("batch_idx,sample_idx,sequence,subdir,target,prediction,confidence\n")
        for sample in sample_info:
            f.write(f"{sample['batch_idx']},{sample['sample_idx']},{sample['sequence']},"
                    f"{sample['subdir']},{sample['target']},{sample['prediction']},{sample['confidence']:.4f}\n")
    
    # Print first 20 predictions for manual verification
    logger.info("===== First 20 Sample Predictions =====")
    logger.info("Idx\tSequence\tSubdir\tGround Truth\tPrediction\tConfidence")
    for i, sample in enumerate(sample_info[:20]):
        logger.info(f"{i}\t{sample['sequence']}\t{sample['subdir']}\t{sample['target']}\t{sample['prediction']}\t{sample['confidence']:.4f}")
    
    # Print metrics
    logger.info("===== Evaluation Metrics =====")
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
    
    # Save metrics to JSON file
    metrics_file = os.path.join(output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump({
            'loss': float(test_loss),
            **serializable_metrics
        }, f, indent=4)
    
    # Compute and save confusion matrix
    class_names = ['No Table', 'Table']
    
    # Plot confusion matrix
    cm_plot_file = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(
        conf_matrix=cm,
        class_names=class_names,
        title='Confusion Matrix',
        save_path=cm_plot_file
    )

    # Plot metrics comparison
    metrics_plot_file = os.path.join(output_dir, 'metrics_comparison.png')
    metrics_to_plot = {
        'Accuracy': test_metrics['accuracy'],
        'Precision': test_metrics['precision'], 
        'Recall': test_metrics['recall'],
        'F1 Score': test_metrics['f1_score'],
        'Specificity': test_metrics['specificity']
    }
    plot_metrics_comparison(
        metrics_dict=metrics_to_plot,
        title="Classification Performance Metrics",
        save_path=metrics_plot_file
    )
    
    # Visualize results with RGB images
    if visualize:
        logger.info(f"Visualizing {len(vis_data)} samples with RGB images...")
        
        for i, sample in enumerate(vis_data):
            # Get RGB image for visualization
            rgb_image = load_rgb_image(
                sequence=sample['sequence'], 
                subdir=sample['subdir'], 
                data_root=dataloader.dataset.dataset.data_root 
                    if hasattr(dataloader.dataset, 'dataset') else 'data/CW2-Dataset/data'
            )
            
            if rgb_image is not None:
                # Set up the visualization path
                rgb_vis_file = os.path.join(vis_dir, f'sample_{i}_rgb_gt_{sample["target"]}_pred_{sample["pred"]}.png')
                
                # Visualize RGB image with classification results
                visualize_classification_results(
                    vis_data,
                    output_dir=rgb_vis_file,
                )
                
                logger.info(f"Saved RGB visualization to {rgb_vis_file}")
            else:
                logger.warning(f"Could not load RGB image for sample {i} from {sample['sequence']}/{sample['subdir']}")
    
    return test_loss, test_metrics

def main(args):
    """
    Main function for evaluating Pipeline A.
    
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
            config['logging']['log_dir'],
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
    test_loss, test_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        output_dir=output_dir,
        num_vis_samples=args.num_vis_samples
    )
    
    logger.info("Evaluation completed!")
    
    # Evaluate on Sun3D dataset
    logger.info("===== Evaluating on Sun3D dataset =====")
    sun3d_loss, sun3d_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        output_dir=os.path.join(output_dir, 'sun3d'),
        visualize=args.visualize,
        num_vis_samples=args.num_vis_samples
    )
    
    # Clear GPU cache before RealSense evaluation
    torch.cuda.empty_cache()

    # Evaluate on RealSense dataset if requested
    if args.eval_realsense:
        logger.info("===== Evaluating on RealSense dataset =====")
        
        # Get RealSense dataloader
        realsense_loader = get_realsense_dataloader(config, pipeline='pipelineA')
        logger.info(f"RealSense dataset loaded with {len(realsense_loader.dataset)} samples")
        
        # Evaluate on RealSense dataset
        realsense_loss, realsense_metrics = evaluate(
            model=model,
            dataloader=realsense_loader,
            criterion=criterion,
            device=device,
            output_dir=os.path.join(output_dir, 'realsense'),
            visualize=args.visualize,
            num_vis_samples=args.num_vis_samples
        )
        
        # Print comparison between datasets
        logger.info("===== Metrics Comparison =====")
        logger.info(f"Metric\t\tSun3D\t\tRealSense")
        logger.info(f"Loss\t\t{sun3d_loss:.4f}\t\t{realsense_loss:.4f}")
        logger.info(f"Accuracy\t{sun3d_metrics['accuracy']:.4f}\t\t{realsense_metrics['accuracy']:.4f}")
        logger.info(f"Precision\t{sun3d_metrics['precision']:.4f}\t\t{realsense_metrics['precision']:.4f}")
        logger.info(f"Recall\t\t{sun3d_metrics['recall']:.4f}\t\t{realsense_metrics['recall']:.4f}")
        logger.info(f"F1 Score\t{sun3d_metrics['f1_score']:.4f}\t\t{realsense_metrics['f1_score']:.4f}")
        logger.info(f"Specificity\t{sun3d_metrics['specificity']:.4f}\t\t{realsense_metrics['specificity']:.4f}")
        
        # Save comparison results
        comparison_file = os.path.join(output_dir, 'dataset_comparison.json')
        with open(comparison_file, 'w') as f:
            json.dump({
                'sun3d': {
                    'loss': float(sun3d_loss),
                    **{k: float(v) for k, v in sun3d_metrics.items()}
                },
                'realsense': {
                    'loss': float(realsense_loss),
                    **{k: float(v) for k, v in realsense_metrics.items()}
                }
            }, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Pipeline A: Point Cloud Classification')
    parser.add_argument('--config', type=str, default='src/pipelineA/config.yaml', help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--num_vis_samples', type=int, default=20, help='Number of samples to visualize')
    parser.add_argument('--eval_realsense', default=True, action='store_true', help='Evaluate on RealSense dataset')
    parser.add_argument('--visualize', default=True, action='store_true', help='Visualize results')
    args = parser.parse_args()
    
    main(args) 