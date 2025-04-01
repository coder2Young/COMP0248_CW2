import os
import numpy as np
import torch
import pandas as pd
import cv2
from torch.utils.data import Dataset
import glob

class RealSenseDataset(Dataset):
    """
    Dataset for RealSense depth camera data.
    """
    def __init__(self, data_root="data/RealSense", num_points=1024, use_height=False, use_rgb=True, transform=None):
        """
        Initialize the RealSense dataset.
        
        Args:
            data_root (str): Root directory containing the RealSense data
            num_points (int): Number of points in the point cloud
            use_height (bool): Whether to use height as an additional feature
            use_rgb (bool): Whether to use RGB as additional features
            transform (callable, optional): Transform to apply to the data
        """
        self.data_root = data_root
        self.num_points = num_points
        self.use_height = use_height
        self.use_rgb = use_rgb
        self.transform = transform
        
        # Define directory paths
        self.depth_dir = os.path.join(data_root, "depthTSDF")
        self.rgb_dir = os.path.join(data_root, "image")
        self.label_csv = os.path.join(data_root, "label.csv")
        
        # Check if directories and label file exist
        if not os.path.exists(self.depth_dir):
            print(f"Warning: Depth directory not found: {self.depth_dir}")
        
        if not os.path.exists(self.rgb_dir) and self.use_rgb:
            print(f"Warning: RGB directory not found: {self.rgb_dir}")
        
        if not os.path.exists(self.label_csv):
            print(f"Warning: Label file not found: {self.label_csv}")
            self.labels = {}
        else:
            # Load labels from CSV
            self.label_df = pd.read_csv(self.label_csv)
            # Convert to dictionary for easy lookup
            self.labels = dict(zip(self.label_df['filename'], self.label_df['label']))
        
        # Find all depth images
        self.depth_files = sorted(glob.glob(os.path.join(self.depth_dir, "*.png")))
        
        # Filter files to only include those with labels
        valid_files = []
        for depth_file in self.depth_files:
            filename = os.path.basename(depth_file)
            
            # Check if there is a label for this file
            if filename in self.labels:
                # If using RGB, also check that RGB file exists
                if self.use_rgb:
                    rgb_file = os.path.join(self.rgb_dir, filename)
                    if os.path.exists(rgb_file):
                        valid_files.append(depth_file)
                else:
                    valid_files.append(depth_file)
        
        self.depth_files = valid_files
        print(f"Found {len(self.depth_files)} RealSense samples")
        
        # Camera intrinsics for RealSense
        self.intrinsics = {
            'fx': 612.7910766601562,
            'fy': 611.8779296875,
            'cx': 321.7364196777344,
            'cy': 245.0658416748047
        }
    
    def __len__(self):
        return len(self.depth_files)
    
    def __getitem__(self, idx):
        # Get file paths
        depth_file = self.depth_files[idx]
        filename = os.path.basename(depth_file)
        rgb_file = os.path.join(self.rgb_dir, filename) if self.use_rgb else None
        
        # Load depth
        depth_map = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
        
        # Convert to meters if needed
        depth_map = depth_map.astype(np.float32)
        if depth_map.max() > 1000:  # If values are in millimeters
            depth_map /= 1000.0  # Convert to meters
        
        # Apply simple noise filtering
        depth_map = cv2.medianBlur(depth_map, 5)
        
        # Load RGB if needed
        if self.use_rgb:
            rgb_image = cv2.imread(rgb_file)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        # Generate point cloud
        xyz_points = []
        rgb_values = []
        
        height, width = depth_map.shape
        fx, fy = self.intrinsics['fx'], self.intrinsics['fy']
        cx, cy = self.intrinsics['cx'], self.intrinsics['cy']
        
        for v in range(0, height, 1):
            for u in range(0, width, 1):
                z = depth_map[v, u]
                if z > 0:
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    xyz_points.append([x, y, z])
                    
                    # Add RGB values if needed
                    if self.use_rgb:
                        # Normalize RGB values to [0, 1]
                        rgb = rgb_image[v, u] / 255.0
                        rgb_values.append(rgb)
        
        if not xyz_points:
            print(f"Warning: No valid depth values found in {depth_file}, returning zero point cloud")
            point_cloud = np.zeros((self.num_points, 6 if self.use_rgb else 3), dtype=np.float32)
        else:
            xyz_points = np.array(xyz_points, dtype=np.float32)
            
            if self.use_rgb:
                rgb_values = np.array(rgb_values, dtype=np.float32)
                point_cloud = np.column_stack((xyz_points, rgb_values))
            else:
                point_cloud = xyz_points
            
            # Subsample or pad to num_points
            if len(point_cloud) > self.num_points:
                indices = np.random.choice(len(point_cloud), self.num_points, replace=False)
                point_cloud = point_cloud[indices]
            elif len(point_cloud) < self.num_points:
                padding = np.zeros((self.num_points - len(point_cloud), 
                                     point_cloud.shape[1]), dtype=np.float32)
                point_cloud = np.vstack([point_cloud, padding])
        
        # Add height feature if needed
        if self.use_height:
            # Extract XYZ coordinates
            xyz = point_cloud[:, :3]
            
            # Calculate height feature (distance from the lowest point in Y)
            floor_height = np.min(xyz[:, 1])
            heights = xyz[:, 1] - floor_height
            
            if self.use_rgb:
                # For 6-channel point cloud: insert height before RGB
                rgb = point_cloud[:, 3:]
                point_cloud = np.column_stack((xyz, heights, rgb))
            else:
                # For 3-channel point cloud: append height
                point_cloud = np.column_stack((xyz, heights))
        
        # Get label for this file
        label = int(self.labels[filename])
        
        # Create sample
        sample = {
            'point_cloud': torch.from_numpy(point_cloud.astype(np.float32)),
            'label': torch.tensor(label, dtype=torch.long)
        }
        
        # Apply transform if provided
        if self.transform:
            sample = self.transform(sample)
        
        return sample

def get_realsense_dataloader(config, pipeline='pipelineA'):
    """
    Get dataloader for RealSense data.
    
    Args:
        config (dict): Configuration dictionary
        pipeline (str): Pipeline name ('pipelineA' or 'pipelineB')
    
    Returns:
        torch.utils.data.DataLoader: DataLoader for RealSense data
    """
    # Create the dataset with the correct data root
    dataset = RealSenseDataset(
        data_root="data/RealSense",  # 使用正确的数据路径
        num_points=config['data']['num_points'],
        use_height=config['data'].get('use_height', False),
        use_rgb=config['data'].get('use_rgb', False),  # 使用配置中的设置
        transform=None  # No transform for evaluation
    )
    
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    return dataloader 