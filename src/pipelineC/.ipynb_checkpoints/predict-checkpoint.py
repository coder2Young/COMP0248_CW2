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
            intrinsics = data['intrinsics']
            
            # if inputs.shape[1] == config['data']['num_points']:  # If second dimension is num_points, we need to transpose
            inputs = inputs.transpose(1, 2)  # (B, N, C) -> (B, C, N)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate predictions
            preds = torch.argmax(outputs, dim=1)  # [B, point_num]
            
            output_dir = 'results/pipelineC/pred'
            os.makedirs(output_dir, exist_ok=True)

            for b in range(inputs.shape[0]):
                points_batch = inputs[b].detach().to("cpu")  # [C, N]
                pred_batch = preds[b].detach().to("cpu")  # [N]

                img = np.ones((480, 640, 3), dtype=np.uint8) * 255

                for n in range(points_batch.shape[1]):
                    x, y, z = points_batch[:3, n]
                    if z == 0:
                        continue
                    # print(f"({x}, {y}, {z})")
                    u = int(x * intrinsics['fx'] / z + intrinsics['cx'])
                    v = int(y * intrinsics['fy'] / z + intrinsics['cy'])

                    if 0 <= u < 640 and 0 <= v < 480:
                        # print(n, pred_batch[n], v, u)
                        if pred_batch[n] == 0:
                            color = (255, 0, 0)
                        elif pred_batch[n] == 1:
                            color = (0, 0, 255)
                        else:
                            color = (255, 255, 255)
                        img[v, u] = color

                img_filename = os.path.join(output_dir, f"{data['file_name'][b].split('/')[-1]}")
                cv2.imwrite(img_filename, img)
                print(f"Saved image {img_filename}")

            
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
    evaluate(
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