import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import glob
from PIL import Image
import random
import open3d as o3d
import matplotlib.pyplot as plt

from src.data_utils.preprocessing import (
    read_intrinsics,
    read_and_parse_polygon_labels,
    depth_to_point_cloud,
    get_image_label_from_polygons,
    depth_to_colored_point_cloud,
    point_cloud_to_depth
)

class Sun3DBaseDataset(Dataset):
    """
    Base dataset class for Sun3D data.
    """
    def __init__(self, data_root, sequences, split='train', transform=None):
        """
        Initialize the Sun3D base dataset.
        
        Args:
            data_root (str): Root directory of the dataset
            sequences (list): List of sequences to include
            split (str): 'train' or 'test'
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_root = data_root
        self.sequences = sequences
        self.split = split
        self.transform = transform
        
        # Sequences known to be all negative samples (no tables)
        self.all_negative_sequences = ["mit_gym_z_squash", "harvard_tea_2"]
        
        # Load all data pairs (RGB, depth, intrinsics, annotations)
        self.data_pairs = self._load_data_pairs()
    
    def _load_data_pairs(self):
        """
        Load all data pairs from the specified sequences.
        
        Returns:
            list: List of dictionaries containing data pairs
        """
        data_pairs = []
        
        # Have you thought about how to handle mismatches between timestamps when pairing RGB and depth images?
        # We will match RGB and depth images by finding the closest timestamp match
        
        for sequence in self.sequences:
            sequence_path = os.path.join(self.data_root, sequence)
            
            # Find all subdirectories in the sequence folder
            subdirs = [d for d in os.listdir(sequence_path) if os.path.isdir(os.path.join(sequence_path, d))]
            
            # Debug counter for total table annotations
            sequence_table_count = 0
            sequence_image_count = 0
            
            for subdir in subdirs:
                subdir_path = os.path.join(sequence_path, subdir)
                
                # Check if the required directories exist
                image_dir = os.path.join(subdir_path, 'image')
                
                # Check for different depth directory names
                depth_dir = None
                possible_depth_dirs = ['depthTSDF', 'depth']
                for possible_dir in possible_depth_dirs:
                    if os.path.exists(os.path.join(subdir_path, possible_dir)):
                        depth_dir = os.path.join(subdir_path, possible_dir)
                        break
                
                labels_dir = os.path.join(subdir_path, 'labels')
                intrinsics_file = os.path.join(subdir_path, 'intrinsics.txt')
                
                # Skip if required directories don't exist
                if not os.path.exists(image_dir) or depth_dir is None:
                    continue
                
                # Read intrinsics
                intrinsics = None
                if os.path.exists(intrinsics_file):
                    intrinsics = read_intrinsics(intrinsics_file)
                
                # Read annotations if available
                annotations = {}
                labels_file = os.path.join(labels_dir, 'tabletop_labels.dat')
                
                # Check if sequence is in all_negative_sequences list or if labels file exists
                if sequence in self.all_negative_sequences:
                    # For sequences known to be all negative, use empty annotations
                    #print(f"Sequence {sequence} is known to have all negative samples (no tables)")
                    annotations = {}  # Empty annotations for all images
                elif os.path.exists(labels_file):
                    annotations = read_and_parse_polygon_labels(labels_file)
        
                # Count images with table annotations but don't print
                subdir_table_count = sum(1 for polygons in annotations.values() if polygons)
                sequence_table_count += subdir_table_count
                
                # Get all RGB images
                rgb_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
                
                # Get all depth maps
                depth_files = sorted(glob.glob(os.path.join(depth_dir, '*.png')))
                
                sequence_image_count += len(rgb_files)
                
                # Parse timestamps from filenames
                rgb_timestamps = {self._get_timestamp_from_filename(os.path.basename(f)): f for f in rgb_files}
                depth_timestamps = {self._get_timestamp_from_filename(os.path.basename(f)): f for f in depth_files}
                
                # Match RGB and depth images by finding the closest timestamp
                for rgb_ts, rgb_file in rgb_timestamps.items():
                    # Find the closest depth timestamp
                    if not depth_timestamps:
                        continue
                    
                    closest_depth_ts = min(depth_timestamps.keys(), key=lambda ts: abs(ts - rgb_ts))
                    depth_file = depth_timestamps[closest_depth_ts]
                    
                    # Get image filename without path
                    rgb_filename = os.path.basename(rgb_file).split('.')[0]
                    
                    # Check if annotations exist for this image
                    # For all_negative_sequences, this will always be an empty list
                    image_annotations = annotations.get(rgb_filename, [])
                    has_table = len(image_annotations) > 0
                   
                    # Add the pair to the list
                    data_pair = {
                        'rgb_file': rgb_file,
                        'depth_file': depth_file,
                        'intrinsics': intrinsics,
                        'annotations': image_annotations,
                        'rgb_timestamp': rgb_ts,
                        'depth_timestamp': closest_depth_ts,
                        'sequence': sequence,
                        'subdir': subdir,
                        'has_table': has_table  # Add a direct flag for debugging
                    }
                    
                    data_pairs.append(data_pair)
        
        # Print the number of sample has table and no table
        if (self.split == "train" or self.split == "val"):
            print("Train Datset:")
        else:
            print("Test Datset:")
        print(f"Number of sample has table: {sum(1 for dp in data_pairs if dp['has_table'])}")
        print(f"Number of sample no table: {sum(1 for dp in data_pairs if not dp['has_table'])}")

        return data_pairs
    
    def _get_timestamp_from_filename(self, filename):
        """
        Extract timestamp from filename.
        
        Args:
            filename (str): Filename
        
        Returns:
            float: Timestamp in seconds
        """
        try:
            # Assuming format like '0000001-000000000000.jpg' or '0000001-000000000000.png'
            # where the second part is the timestamp in microseconds
            parts = filename.split('-')
            if len(parts) >= 2:
                timestamp_str = parts[1].split('.')[0]
                return float(timestamp_str) / 1e6  # Convert microseconds to seconds
            return 0.0
        except:
            return 0.0
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        """
        Get a data sample.
        
        Args:
            idx (int): Index
        
        Returns:
            dict: A data sample
        """
        data_pair = self.data_pairs[idx]
        
        # Load RGB image
        rgb_image = cv2.imread(data_pair['rgb_file'])
        if rgb_image is not None:
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Load depth map
        depth_map = cv2.imread(data_pair['depth_file'], cv2.IMREAD_ANYDEPTH)
        if depth_map is None:
            depth_map = np.zeros((480, 640), dtype=np.uint16)
        
        # Convert depth map to meters if needed
        depth_map = depth_map.astype(np.float32)
        if depth_map.max() > 1000:  # If values are in millimeters
            #print("Depth map unit is millimeters, converting to meters")
            depth_map /= 1000.0  # Convert to meters
        
        # Apply a simple median filter to reduce noise
        depth_map = cv2.medianBlur(depth_map.astype(np.float32), 5)
        
        # Get image shape
        image_shape = rgb_image.shape[:2]
        
        # Get intrinsics - use default values if intrinsics are missing
        intrinsics = data_pair['intrinsics']
        if intrinsics is None:
            # Use default intrinsics as fallback
            intrinsics = {
                'fx': 525.0,  # Default values from common RGBD cameras
                'fy': 525.0,
                'cx': 319.5,
                'cy': 239.5
            }
            print(f"Warning: Using default intrinsics for {data_pair['rgb_file']}")
        
        # Get annotations
        annotations = data_pair['annotations']
        
        # Get binary label and mask
        binary_label, binary_mask = get_image_label_from_polygons(annotations, image_shape)
        
        # Create sample dictionary
        sample = {
            'rgb_image': rgb_image,
            'depth_map': depth_map,
            'intrinsics': intrinsics,
            'annotations': annotations,
            'binary_label': binary_label,
            'binary_mask': binary_mask,
            'sequence': data_pair['sequence'],
            'subdir': data_pair['subdir']
        }
        
        # Note: We don't apply transform here since it will be handled by child classes
        # The child classes can apply transformations after adding their specific data
        
        return sample

class TablePointCloudDataset(Sun3DBaseDataset):
    """
    Sun3D dataset for point cloud classification and segmentation.
    """
    def __init__(self, data_root, sequences, split='train', mode='classification', 
                 num_points=1024, transform=None, use_rgb=True):
        """
        Initialize the point cloud dataset.
        
        Args:
            data_root (str): Root directory of the dataset
            sequences (list): List of sequences to use
            split (str): 'train', 'val', or 'test' split
            mode (str): 'classification' or 'segmentation'
            num_points (int): Number of points in the point cloud
            transform (callable, optional): Transform to apply to the data
            use_rgb (bool): Whether to use RGB as additional features
        """
        super().__init__(data_root, sequences, split, transform)
        self.mode = mode
        self.num_points = num_points
        self.use_rgb = use_rgb
    
    def __getitem__(self, idx):
        """
        Get a data sample.
        
        Args:
            idx (int): Index
        
        Returns:
            dict: A data sample including point cloud and labels
        """
        # Get base sample from Sun3DBaseDataset
        sample = super().__getitem__(idx)
        
        # Get depth map and intrinsics
        depth_map = sample['depth_map']
        intrinsics = sample['intrinsics']
        
        # Generate point cloud with RGB
        if self.use_rgb:
            point_cloud = depth_to_colored_point_cloud(
                depth_map, 
                sample['rgb_image'],
                intrinsics, 
                subsample=True, 
                num_points=self.num_points
            )
        else:
            point_cloud = depth_to_point_cloud(
                depth_map, 
                intrinsics, 
                subsample=True, 
                num_points=self.num_points
            )
        
        # Add point cloud to the sample
        sample['point_cloud'] = point_cloud.astype(np.float32)
        
        # Now apply any transforms AFTER creating the point cloud
        # The transform is already applied in the parent class for other data,
        # but we need to apply it again after adding the point cloud
        if self.transform:
            sample = self.transform(sample)
        
        # Create a new dictionary for the processed sample (pc_sample)
        pc_sample = {}
        
        # Pass through sequence and subdir information for debugging
        pc_sample['sequence'] = sample['sequence']
        pc_sample['subdir'] = sample['subdir']
        
        # For classification, we just need the point cloud and the binary label
        if self.mode == 'classification':
            pc_sample['point_cloud'] = sample['point_cloud']  # Already a numpy array
            pc_sample['label'] = sample['binary_label']
        
        # For segmentation, we need point-level labels
        else:  # self.mode == 'segmentation'
            # Project binary mask to point cloud
            binary_mask = sample['binary_mask']
            
            # Initialize point-level labels
            point_labels = np.zeros(self.num_points, dtype=np.int64)
            
            # Get image dimensions
            height, width = depth_map.shape
            
            # Use the point cloud from the sample (may have been transformed)
            curr_points = sample['point_cloud']
            
            # For each point in the point cloud, check if it belongs to a table
            for i, (x, y, z) in enumerate(curr_points[:, :3]):  # Only use XYZ coordinates
                if z <= 0:
                    continue
                
                # Project the point back to image space
                u = int(x * intrinsics['fx'] / z + intrinsics['cx'])
                v = int(y * intrinsics['fy'] / z + intrinsics['cy'])
                
                # Check if the projected point is inside the image and has a valid mask value
                if 0 <= u < width and 0 <= v < height:
                    point_labels[i] = binary_mask[v, u]
            
            pc_sample['point_cloud'] = sample['point_cloud']
            pc_sample['labels'] = point_labels
        
        # Convert to PyTorch tensors
        for key in pc_sample:
            if isinstance(pc_sample[key], np.ndarray):
                pc_sample[key] = torch.from_numpy(pc_sample[key].astype(np.float32))
        
        return pc_sample

class DatasetSplitter:
    """
    Utility class to split sequences into train and test sets.
    """
    @staticmethod
    def get_train_test_sequences():
        """
        Get train and test sequences based on the course requirements.
        
        Returns:
            tuple: (train_sequences, test_sequences)
        """
        # Define train and test sequences as specified
        train_sequences = [
            'mit_32_d507',
            'mit_76_459', 
            'mit_76_studyroom',
            'mit_gym_z_squash',
            'mit_lab_hj'
        ]
        
        test_sequences = [
            'harvard_c5',
            'harvard_c6',
            'harvard_c11',
            'harvard_tea_2'
        ]
        
        return train_sequences, test_sequences 