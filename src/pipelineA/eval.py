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
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from src.pipelineA.model import get_model
from src.pipelineA.dataset import get_dataloaders, load_config
from src.utils.metrics import compute_classification_metrics
from src.utils.visualization import (
    visualize_point_cloud, 
    plot_confusion_matrix, 
    visualize_classification_results,
    plot_metrics_comparison
)

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
    
    # Sample info for debugging
    sample_info = []
    
    # Create directories for outputs
    os.makedirs(output_dir, exist_ok=True)
    if visualize:
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    # Sample data for visualization
    vis_data = []
    
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
                    # Try to get the RGB image if available (it's in the dataset but might not be in the dataloader)
                    # We'll need to retrieve it when visualizing
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
    
    # Print label distribution
    target_counts = np.bincount(np.array(all_targets), minlength=2)
    pred_counts = np.bincount(np.array(all_preds), minlength=2)
    
    print("\n===== Label Distribution Statistics =====")
    print(f"Ground Truth: No Table (0): {target_counts[0]}, Table (1): {target_counts[1]}")
    print(f"Predictions: No Table (0): {pred_counts[0]}, Table (1): {pred_counts[1]}")
    print("========================================\n")
    
    # Save detailed sample info to CSV
    sample_info_file = os.path.join(output_dir, 'sample_predictions.csv')
    with open(sample_info_file, 'w') as f:
        f.write("batch_idx,sample_idx,sequence,subdir,target,prediction,confidence\n")
        for sample in sample_info:
            f.write(f"{sample['batch_idx']},{sample['sample_idx']},{sample['sequence']},"
                    f"{sample['subdir']},{sample['target']},{sample['prediction']},{sample['confidence']:.4f}\n")
    
    # Print first 20 predictions for manual verification
    print("\n===== First 20 Sample Predictions =====")
    print("Idx\tSequence\tSubdir\tGround Truth\tPrediction\tConfidence")
    for i, sample in enumerate(sample_info[:20]):
        print(f"{i}\t{sample['sequence']}\t{sample['subdir']}\t{sample['target']}\t{sample['prediction']}\t{sample['confidence']:.4f}")
    print("=======================================\n")
    
    # Print metrics
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"Test Specificity: {test_metrics['specificity']:.4f}")
    
    # Print confusion matrix for better visualization
    cm = confusion_matrix(all_targets, all_preds)
    print("\n===== Confusion Matrix =====")
    print("GT\\Pred\tNo Table(0)\tTable(1)")
    print(f"No Table(0)\t{cm[0][0]}\t\t{cm[0][1]}")
    print(f"Table(1)\t{cm[1][0]}\t\t{cm[1][1]}")
    print("===========================\n")
    
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
    
    # Visualize results
    if visualize:
        # Function to load RGB image for visualization
        def load_rgb_image(sequence, subdir, data_root='data/CW2-Dataset/data'):
            # This is just a helper to try to find the RGB images for the test samples
            # It's not guaranteed to work for all datasets, but it's a best effort
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
            
        for i, sample in enumerate(vis_data):
            # First, visualize the point cloud
            pc_vis_file = os.path.join(output_dir, 'visualizations', f'sample_{i}_pointcloud_gt_{sample["target"]}_pred_{sample["pred"]}.png')
            
            visualize_point_cloud(
                point_cloud=sample['point_cloud'],
                title=f'Ground Truth: {class_names[sample["target"]]}, '
                      f'Predicted: {class_names[sample["pred"]]} '
                      f'(Conf: {sample["confidence"]:.2f})',
                save_path=pc_vis_file
            )
            
            # Then, try to visualize the RGB image with prediction
            rgb_image = load_rgb_image(sample['sequence'], sample['subdir'])
            if rgb_image is not None:
                rgb_vis_file = os.path.join(output_dir, 'visualizations', f'sample_{i}_rgb_gt_{sample["target"]}_pred_{sample["pred"]}.png')
                
                # Visualize RGB image with classification results
                visualize_classification_results(
                    rgb_image=rgb_image,
                    ground_truth_label=sample['target'],
                    predicted_label=sample['pred'],
                    title=f'Sample {i} - {sample["sequence"]}/{sample["subdir"]}',
                    class_names=class_names,
                    confidence=sample['confidence'],
                    save_path=rgb_vis_file
                )
    
    return test_loss, test_metrics

def main(args):
    """
    Main function for evaluating Pipeline A.
    
    Args:
        args: Command line arguments
    """
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get dataloaders
    _, _, test_loader = get_dataloaders(args.config)
    
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
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create output directory
    output_dir = os.path.join(config['logging']['log_dir'], 'evaluation')
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    print("Evaluation completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Pipeline A: Point Cloud Classification')
    parser.add_argument('--config', type=str, default='src/pipelineA/config.yaml', help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    parser.add_argument('--num_vis_samples', type=int, default=10, help='Number of samples to visualize')
    args = parser.parse_args()
    
    main(args) 