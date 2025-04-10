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

def visualize_classification_results(vis_data, output_dir, class_names=None):
    """
    Visualizes classification results, including RGB, predicted depth, and optionally GT depth.

    Args:
        vis_data (list): List of dictionaries, each containing sample data like
                         'rgb_tensor', 'pred_depth', 'gt_depth' (can be None), 'target', 'pred'.
        output_dir (str): Directory to save the visualization images.
        class_names (list, optional): List of class names. Defaults to ['No Table', 'Table'].
    """
    if class_names is None:
        class_names = ['No Table', 'Table']
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {len(vis_data)} classification visualizations in {output_dir}...")

    for i, sample in enumerate(vis_data):
        try:
            rgb_tensor = sample.get('rgb_tensor')
            pred_depth_tensor = sample.get('pred_depth')
            gt_depth_tensor = sample.get('gt_depth') # This can be None or a Tensor
            target = sample.get('target', -1)
            pred = sample.get('pred', -1)
            confidence = sample.get('confidence', -1.0)

            # --- Data Conversion ---
            # Convert RGB
            if rgb_tensor is None or not isinstance(rgb_tensor, torch.Tensor):
                 print(f"Warning: Skipping sample {i}, missing or invalid RGB tensor.")
                 continue
            rgb_np = rgb_tensor.numpy().transpose(1, 2, 0)
            rgb_np = np.clip(rgb_np, 0, 1)

            # Convert Predicted Depth
            if pred_depth_tensor is None or not isinstance(pred_depth_tensor, torch.Tensor):
                 print(f"Warning: Skipping sample {i}, missing or invalid predicted depth tensor.")
                 continue
            pred_depth_np = pred_depth_tensor.numpy()

            # Convert GT Depth (Handle None)
            gt_depth_np = None
            if gt_depth_tensor is not None and isinstance(gt_depth_tensor, torch.Tensor):
                gt_depth_np = gt_depth_tensor.numpy()
            # --- End Data Conversion ---


            # --- Plotting ---
            fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # Adjusted size slightly
            fig.suptitle(f"Classification Vis #{i} - Target: {class_names[target]}, Pred: {class_names[pred]} ({confidence:.2f})")

            # Plot RGB
            axes[0].imshow(rgb_np)
            axes[0].set_title("RGB Image")
            axes[0].axis('off')

            # Plot Predicted Depth
            im1 = axes[1].imshow(pred_depth_np, cmap='viridis')
            axes[1].set_title("Predicted Depth")
            axes[1].axis('off')
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            # Plot Ground Truth Depth (Check for None!)
            ax2 = axes[2]
            if gt_depth_np is not None:
                im2 = ax2.imshow(gt_depth_np, cmap='viridis') # Only plot if not None
                ax2.set_title("Ground Truth Depth")
                fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            else:
                # Display placeholder text if GT depth is None
                ax2.text(0.5, 0.5, 'GT Depth N/A', horizontalalignment='center',
                       verticalalignment='center', transform=ax2.transAxes, fontsize=12, color='gray')
                ax2.set_title("Ground Truth Depth")
            ax2.axis('off')
            # --- End Plotting ---


            # Add overall correctness indicator text below plots if needed
            # ... (optional text)

            # Save the figure
            filename = f"classification_vis_{sample.get('batch_idx', 'X')}_{sample.get('sample_idx', i)}.png"
            save_path = os.path.join(output_dir, filename)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(save_path, dpi=100)
            plt.close(fig)

        except Exception as plot_err:
            print(f"Warning: Failed to generate visualization for sample {i}. Error: {plot_err}")
            if 'fig' in locals() and plt.fignum_exists(fig.number): # Close figure if error occurred mid-plot
                 plt.close(fig)

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

def visualize_point_cloud_segmentation(point_cloud, point_colors=None, target_labels=None, pred_labels=None, 
                                    rgb_image=None, title='Point Cloud Segmentation', save_path=None, figsize=(15, 8)):
    """
    Visualize segmentation results for point clouds with optional RGB image.
    
    Args:
        point_cloud (numpy.ndarray): Point cloud of shape (N, 3)
        point_colors (numpy.ndarray, optional): Point colors of shape (N, 3)
        target_labels (numpy.ndarray, optional): Target labels of shape (N,)
        pred_labels (numpy.ndarray, optional): Predicted labels of shape (N,)
        rgb_image (numpy.ndarray, optional): RGB image to display alongside the point cloud
        title (str): Title for the plot
        save_path (str, optional): Path to save the visualization
        figsize (tuple): Figure size (width, height)
    """
    # Convert to numpy array if needed
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.detach().cpu().numpy()
    
    if point_colors is not None and isinstance(point_colors, torch.Tensor):
        point_colors = point_colors.detach().cpu().numpy()
    
    if target_labels is not None and isinstance(target_labels, torch.Tensor):
        target_labels = target_labels.detach().cpu().numpy()
    
    if pred_labels is not None and isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.detach().cpu().numpy()
    
    # Determine the number of plots needed
    has_rgb = rgb_image is not None
    has_pred = pred_labels is not None
    
    if has_rgb and has_pred:
        # RGB + Target + Prediction
        n_plots = 3
    elif has_rgb or has_pred:
        # RGB + Target or Target + Prediction
        n_plots = 2
    else:
        # Target only
        n_plots = 1
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Add RGB image if available
    plot_idx = 1
    if has_rgb:
        ax_rgb = fig.add_subplot(1, n_plots, plot_idx)
        ax_rgb.imshow(rgb_image)
        ax_rgb.set_title('RGB Image')
        ax_rgb.axis('off')
        plot_idx += 1
    
    # Add target point cloud
    
    # Define colors for target labels
    if target_labels is not None:
        target_colors = np.zeros((len(point_cloud), 3))
        target_colors[target_labels == 0] = [0, 0, 1]  # Blue for background
        target_colors[target_labels == 1] = [1, 0, 0]  # Red for table
    elif point_colors is not None:
        # Use provided point colors
        target_colors = point_colors
    else:
        # Use a default color
        target_colors = np.array([[0, 0.651, 0.929]] * len(point_cloud))  # Azure
    
    # Plot target point cloud
    if target_labels is not None:
        ax_target = fig.add_subplot(1, n_plots, plot_idx, projection='3d')
        ax_target.scatter(
            point_cloud[:, 0],
            point_cloud[:, 1],
            point_cloud[:, 2],
            c=target_colors,
            s=5,  # point size
            alpha=0.8  # transparency
        )

        # Set title and labels
        ax_target.set_title('Ground Truth\nBlue: Background, Red: Table')
        ax_target.set_xlabel('X')
        ax_target.set_ylabel('Y')
        ax_target.set_zlabel('Z')

        # Set equal aspect ratio
        ax_target.set_box_aspect([1, 1, 1])

        # Adjust view angle
        ax_target.view_init(elev=30, azim=45)
    else:
        plot_idx -= 1
    
    # Add prediction point cloud if available
    if has_pred:
        plot_idx += 1
        ax_pred = fig.add_subplot(1, n_plots, plot_idx, projection='3d')
        
        # Define colors for predicted labels
        pred_colors = np.zeros((len(point_cloud), 3))
        pred_colors[pred_labels == 0] = [0, 0, 1]  # Blue for background
        pred_colors[pred_labels == 1] = [1, 0, 0]  # Red for table
        
        # Plot predicted point cloud
        ax_pred.scatter(
            point_cloud[:, 0],
            point_cloud[:, 1],
            point_cloud[:, 2],
            c=pred_colors,
            s=2,  # point size
            alpha=0.8  # transparency
        )
        
        # Set title and labels
        ax_pred.set_title('Prediction\nBlue: Background, Red: Table')
        ax_pred.set_xlabel('X')
        ax_pred.set_ylabel('Y')
        ax_pred.set_zlabel('Z')
        
        # Set equal aspect ratio
        ax_pred.set_box_aspect([1, 1, 1])
        
        # Adjust view angle to match target
        if target_labels is not None:
            ax_pred.view_init(elev=30, azim=45)
        else:
            ax_pred.view_init(elev=-90, azim=-90)
    
    # Set overall title
    plt.suptitle(title, fontsize=16)
    
    # Save figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show() 