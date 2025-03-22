import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os
import open3d as o3d
from matplotlib.colors import ListedColormap

def visualize_point_cloud(point_cloud, labels=None, title='Point Cloud', save_path=None, figsize=(10, 10)):
    """
    Visualize point cloud in 3D.
    
    Args:
        point_cloud (numpy.ndarray or torch.Tensor): Point cloud of shape (N, 3)
        labels (numpy.ndarray or torch.Tensor, optional): Point labels of shape (N,)
        title (str, optional): Title of the plot
        save_path (str, optional): Path to save the visualization
        figsize (tuple, optional): Figure size
    """
    # Convert to numpy array if needed
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.detach().cpu().numpy()
    
    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get colors based on labels if provided
    if labels is not None:
        # Define colors for different classes
        # For binary segmentation, use red for table (1) and blue for background (0)
        colors = np.array(['blue', 'red'])
        point_colors = colors[labels]
    else:
        # Use a single color if no labels are provided
        point_colors = 'blue'
    
    # Plot the points
    ax.scatter(
        point_cloud[:, 0],
        point_cloud[:, 1],
        point_cloud[:, 2],
        c=point_colors,
        s=5,  # point size
        alpha=0.8  # transparency
    )
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Adjust view angle
    ax.view_init(elev=30, azim=45)
    
    # Add a legend if labels are provided
    if labels is not None:
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Background'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Table')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    # Save figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def visualize_point_cloud_open3d(point_cloud, labels=None, title='Point Cloud', save_path=None):
    """
    Visualize point cloud using Open3D.
    
    Args:
        point_cloud (numpy.ndarray or torch.Tensor): Point cloud of shape (N, 3)
        labels (numpy.ndarray or torch.Tensor, optional): Point labels of shape (N,)
        title (str, optional): Title of the visualization window
        save_path (str, optional): Path to save the visualization as a screenshot
    """
    # Convert to numpy array if needed
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.detach().cpu().numpy()
    
    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
    # Set colors based on labels if provided
    if labels is not None:
        # Define colors for different classes
        # For binary segmentation, use red for table (1) and blue for background (0)
        colors = np.zeros((len(point_cloud), 3))
        colors[labels == 0] = [0, 0, 1]  # Blue for background
        colors[labels == 1] = [1, 0, 0]  # Red for table
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Use a single color if no labels are provided
        pcd.paint_uniform_color([0, 0.651, 0.929])  # Azure
    
    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=800, height=600)
    
    # Add the point cloud to the visualizer
    vis.add_geometry(pcd)
    
    # Set view control options
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])  # White background
    opt.point_size = 3.0  # Increase point size
    
    # Update the view
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    
    # Save screenshot if save_path is provided
    if save_path is not None:
        vis.capture_screen_image(save_path)
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()

def visualize_segmentation_results(rgb_image, depth_map, ground_truth_mask, predicted_mask=None, 
                                  title='Segmentation Results', save_path=None, figsize=(15, 10)):
    """
    Visualize segmentation results.
    
    Args:
        rgb_image (numpy.ndarray): RGB image of shape (H, W, 3)
        depth_map (numpy.ndarray): Depth map of shape (H, W)
        ground_truth_mask (numpy.ndarray): Ground truth mask of shape (H, W)
        predicted_mask (numpy.ndarray, optional): Predicted mask of shape (H, W)
        title (str, optional): Title of the plot
        save_path (str, optional): Path to save the visualization
        figsize (tuple, optional): Figure size
    """
    # Convert to numpy array if needed
    if isinstance(rgb_image, torch.Tensor):
        rgb_image = rgb_image.detach().cpu().numpy()
    
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.detach().cpu().numpy()
    
    if isinstance(ground_truth_mask, torch.Tensor):
        ground_truth_mask = ground_truth_mask.detach().cpu().numpy()
    
    if predicted_mask is not None and isinstance(predicted_mask, torch.Tensor):
        predicted_mask = predicted_mask.detach().cpu().numpy()
    
    # Normalize depth map for visualization
    depth_normalized = depth_map.copy()
    if depth_normalized.max() > 0:
        depth_normalized = depth_normalized / depth_normalized.max()
    
    # Apply colormap to depth map
    depth_colormap = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
    
    # Create figure
    if predicted_mask is not None:
        # Create a figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        axs = axs.ravel()
        
        # Plot RGB image
        axs[0].imshow(rgb_image)
        axs[0].set_title('RGB Image')
        axs[0].axis('off')
        
        # Plot depth map
        axs[1].imshow(depth_colormap)
        axs[1].set_title('Depth Map')
        axs[1].axis('off')
        
        # Plot ground truth mask
        axs[2].imshow(ground_truth_mask, cmap='binary')
        axs[2].set_title('Ground Truth Mask')
        axs[2].axis('off')
        
        # Plot predicted mask
        axs[3].imshow(predicted_mask, cmap='binary')
        axs[3].set_title('Predicted Mask')
        axs[3].axis('off')
    else:
        # Create a figure with 1x3 subplots
        fig, axs = plt.subplots(1, 3, figsize=figsize)
        
        # Plot RGB image
        axs[0].imshow(rgb_image)
        axs[0].set_title('RGB Image')
        axs[0].axis('off')
        
        # Plot depth map
        axs[1].imshow(depth_colormap)
        axs[1].set_title('Depth Map')
        axs[1].axis('off')
        
        # Plot ground truth mask
        axs[2].imshow(ground_truth_mask, cmap='binary')
        axs[2].set_title('Ground Truth Mask')
        axs[2].axis('off')
    
    # Set overall title
    plt.suptitle(title)
    
    # Save figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def visualize_classification_results(rgb_image, ground_truth_label=None, predicted_label=None,
                                   title='Classification Results', class_names=None, confidence=None, save_path=None, figsize=(15, 5)):
    """
    Visualize classification results on an RGB image.
    
    Args:
        rgb_image (numpy.ndarray): RGB image to visualize
        ground_truth_label (int, optional): Ground truth label
        predicted_label (int, optional): Predicted label
        title (str): Title for the plot
        class_names (list): List of class names, defaults to ['No Table', 'Table'] if None
        confidence (float): Prediction confidence score (optional)
        save_path (str, optional): Path to save the visualization
        figsize (tuple): Figure size (width, height)
    """
    if class_names is None:
        class_names = ['No Table', 'Table']
        
    # Create figure
    plt.figure(figsize=figsize)
    
    # Show RGB image
    plt.imshow(rgb_image)
    
    # Format title with ground truth and prediction information
    if ground_truth_label is not None and predicted_label is not None:
        gt_class = class_names[ground_truth_label] if ground_truth_label < len(class_names) else str(ground_truth_label)
        pred_class = class_names[predicted_label] if predicted_label < len(class_names) else str(predicted_label)
        
        # Check if prediction is correct
        is_correct = ground_truth_label == predicted_label
        status = "Correct" if is_correct else "Incorrect"
        status_color = "green" if is_correct else "red"
        
        # Add confidence if available
        conf_text = f", Confidence: {confidence:.2f}" if confidence is not None else ""
        
        # Create full title
        full_title = f"{title}\nGround Truth: {gt_class}, Prediction: {pred_class}{conf_text}\n"
        plt.title(full_title)
        
        # Add a text box showing if the prediction was correct or not
        plt.figtext(0.5, 0.01, status, ha="center", fontsize=12, 
                    bbox={"facecolor": status_color, "alpha": 0.5, "pad": 5})
    else:
        plt.title(title)
    
    # Remove axes
    plt.axis('off')
    
    # Tight layout
    plt.tight_layout()
    
    # Save or display figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(conf_matrix, class_names, title='Confusion Matrix', 
                        save_path=None, figsize=(8, 6), cmap=plt.cm.Blues):
    """
    Plot confusion matrix.
    
    Args:
        conf_matrix (numpy.ndarray): Confusion matrix of shape (n_classes, n_classes)
        class_names (list): List of class names
        title (str, optional): Title of the plot
        save_path (str, optional): Path to save the visualization
        figsize (tuple, optional): Figure size
        cmap (matplotlib.colors.Colormap, optional): Colormap for the confusion matrix
    """
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot confusion matrix
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    # Set tick marks and labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Display values in each cell
    thresh = conf_matrix.max() / 2.0
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    
    # Add labels and tight layout
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()

def plot_training_curves(train_metrics, val_metrics, metric_name, title=None, 
                        save_path=None, figsize=(10, 6)):
    """
    Plot training and validation metrics over epochs.
    
    Args:
        train_metrics (list): List of training metrics over epochs
        val_metrics (list): List of validation metrics over epochs
        metric_name (str): Name of the metric
        title (str, optional): Title of the plot
        save_path (str, optional): Path to save the visualization
        figsize (tuple, optional): Figure size
    """
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot training and validation metrics
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'b-', label=f'Training {metric_name}')
    plt.plot(epochs, val_metrics, 'r-', label=f'Validation {metric_name}')
    
    # Set title and labels
    if title is None:
        title = f"{metric_name} over Epochs"
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_metrics_comparison(metrics_dict, title="Performance Metrics", save_path=None, figsize=(10, 6)):
    """
    Plot a bar chart comparing different performance metrics.
    
    Args:
        metrics_dict (dict): Dictionary containing metric names and values
        title (str): Chart title
        save_path (str): Path to save the figure
        figsize (tuple): Figure size (width, height)
    """
    # Create figure
    plt.figure(figsize=figsize)
    
    # Get metric names and values
    labels = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    # Create bar colors - use a blue gradient
    colors = plt.cm.Blues(np.linspace(0.5, 0.9, len(labels)))
    
    # Create horizontal bar chart
    bars = plt.barh(labels, values, color=colors)
    
    # Customize plot
    plt.xlabel('Value')
    plt.ylabel('Metric')
    plt.title(title)
    plt.xlim(0, 1.0)  # Assuming metrics are between 0 and 1
    
    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                 ha='left', va='center')
    
    # Add grid
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    # Save or display figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show() 