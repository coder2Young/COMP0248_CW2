import os
import numpy as np
import torch
import pandas as pd
import cv2
from torch.utils.data import Dataset
from PIL import Image
import open3d as o3d
import torch.utils.data as data
from torchvision import transforms

class RealSenseDataset(Dataset):
    """
    Dataset for RealSense RGB-D images with table/no table classification.
    """
    def __init__(self, data_root='data/RealSense', transform=None, point_transform=None, image_size=384, use_rgb=False):
        """
        Initialize the dataset.
        
        Args:
            data_root (str): Root directory of the dataset
            transform (callable, optional): Transform for RGB images
            point_transform (callable, optional): Transform for point clouds
            image_size (int): Image size for resizing
        """
        self.data_root = data_root
        self.transform = transform
        self.point_transform = point_transform
        self.image_size = image_size
        self.use_rgb = use_rgb
        # Read label CSV file
        label_file = os.path.join(data_root, 'label.csv')
        self.labels_df = pd.read_csv(label_file)
        
        # Get file list
        self.file_list = self.labels_df['filename'].tolist()
        
        # Intrinsic parameters for Intel RealSense depth camera
        # These are default values - ideally should be loaded from calibration file
        self.fx = 616.4  # focal length x
        self.fy = 616.7  # focal length y
        self.cx = 317.6  # principal point x
        self.cy = 242.5  # principal point y
        
        # Verify files exist
        self._verify_files()
    
    def _verify_files(self):
        """Verify all files exist in both image and depth directories."""
        verified_files = []
        
        for filename in self.file_list:
            rgb_path = os.path.join(self.data_root, 'image', filename)
            depth_path = os.path.join(self.data_root, 'depthTSDF', filename)
            
            if os.path.exists(rgb_path) and os.path.exists(depth_path):
                verified_files.append(filename)
            else:
                print(f"Warning: Missing files for {filename}")
        
        # Update file list to only include verified files
        self.file_list = verified_files
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.file_list)
    
    def _load_rgb(self, idx):
        """Load RGB image."""
        filename = self.file_list[idx]
        rgb_path = os.path.join(self.data_root, 'image', filename)
        
        # Load image and convert to RGB
        rgb_image = cv2.imread(rgb_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        if self.image_size:
            rgb_image = cv2.resize(rgb_image, (self.image_size, self.image_size))
        
        return rgb_image
    
    def _load_depth(self, idx):
        """Load depth image."""
        filename = self.file_list[idx]
        depth_path = os.path.join(self.data_root, 'depthTSDF', filename)
        
        # Load depth image
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        # Convert to meters if needed (depends on how depth is stored)
        # Assuming depth is stored in millimeters
        depth_image = depth_image.astype(np.float32) / 1000.0
        
        # Resize if needed
        if self.image_size:
            depth_image = cv2.resize(depth_image, (self.image_size, self.image_size))
        
        return depth_image
    
    def _create_point_cloud(self, rgb_image, depth_image):
        """
        Create point cloud from RGB and depth images.
        
        Args:
            rgb_image (numpy.ndarray): RGB image of shape (H, W, 3)
            depth_image (numpy.ndarray): Depth image of shape (H, W)
            
        Returns:
            numpy.ndarray: Point cloud of shape (N, 6) with XYZ coordinates and RGB values
        """
        # Get image dimensions
        height, width = depth_image.shape
        
        # Create coordinate grid
        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)
        
        # Compute 3D coordinates
        z = depth_image
        x = (xx - self.cx) * z / self.fx
        y = (yy - self.cy) * z / self.fy
        
        # Reshape to point cloud format
        x = x.reshape(-1)
        y = y.reshape(-1)
        z = z.reshape(-1)
        
        # Get RGB values
        r = rgb_image[:, :, 0].reshape(-1)
        g = rgb_image[:, :, 1].reshape(-1)
        b = rgb_image[:, :, 2].reshape(-1)
        
        # Filter out invalid points (zero depth)
        valid_idx = z > 0
        x = x[valid_idx]
        y = y[valid_idx]
        z = z[valid_idx]
        r = r[valid_idx]
        g = g[valid_idx]
        b = b[valid_idx]
        
        # Combine into point cloud
        if self.use_rgb:
            points = np.column_stack([x, y, z, r, g, b])
        else:
            points = np.column_stack([x, y, z])
        
        # Return the first 10000 points for compatibility (or sample if more)
        if len(points) > 10000:
            indices = np.random.choice(len(points), 10000, replace=False)
            points = points[indices]
        elif len(points) < 10000:
            # Pad with zeros if not enough points
            padding = np.zeros((10000 - len(points), 6))
            points = np.vstack([points, padding])
        
        return points
    
    def get_label(self, idx):
        """Get label for a sample."""
        filename = self.file_list[idx]
        # Find the corresponding row in the dataframe
        label = self.labels_df.loc[self.labels_df['filename'] == filename, 'label'].values[0]
        return int(label)
    
    def __getitem__(self, idx):
        """
        Get a data sample.
        
        Args:
            idx (int): Index
            
        Returns:
            dict: Sample dictionary with RGB image, depth map, point cloud, and label
        """
        # Load RGB and depth images
        rgb_image = self._load_rgb(idx)
        depth_image = self._load_depth(idx)
        
        # Get label
        label = self.get_label(idx)
        
        # Create point cloud for PipelineA
        point_cloud = self._create_point_cloud(rgb_image, depth_image)
        
        # Convert to tensors
        rgb_tensor = torch.from_numpy(rgb_image.transpose(2, 0, 1).astype(np.float32) / 255.0)
        depth_tensor = torch.from_numpy(depth_image.astype(np.float32))
        point_cloud_tensor = torch.from_numpy(point_cloud.astype(np.float32))
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # Apply transforms if provided
        if self.transform is not None:
            rgb_tensor = self.transform(rgb_tensor)
        
        if self.point_transform is not None:
            point_cloud_tensor = self.point_transform(point_cloud_tensor)
        
        # Create sample dictionary
        sample = {
            'rgb_image': rgb_tensor,
            'depth_map': depth_tensor,
            'point_cloud': point_cloud_tensor,
            'label': label_tensor,
            'file_path': self.file_list[idx],
            'dataset': 'realsense'  # Marker to identify the dataset source
        }
        
        return sample

def get_realsense_dataloader(config, pipeline='pipelineB'):
    """
    Get dataloader for RealSense dataset.
    
    Args:
        config (dict): Configuration dictionary
        pipeline (str): Pipeline name ('pipelineA' or 'pipelineB')
        
    Returns:
        torch.utils.data.DataLoader: Test dataloader
    """

    
    # Set up appropriate transforms based on pipeline
    if pipeline == 'pipelineA':
        # For point cloud classification
        transform = None
        point_transform = None  # Define point transform if needed
    else:
        # For RGB-based classification (PipelineB)
        transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        point_transform = None
    
    # Get image size from config
    image_size = config['data'].get('image_size', 384)
    
    # Create dataset
    dataset = RealSenseDataset(
        data_root=config.get('realsense_data_root', 'data/RealSense'),
        transform=transform,
        point_transform=point_transform,
        image_size=image_size
    )
    
    # Create dataloader
    dataloader = data.DataLoader(
        dataset,
        batch_size=config['data'].get('batch_size', 32),
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True
    )
    
    return dataloader 