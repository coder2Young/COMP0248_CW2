import torch
import torch.utils.data as data
import numpy as np
import os
import cv2
import random
from torchvision import transforms
from PIL import Image

from src.data_utils.dataset import Sun3DBaseDataset, DatasetSplitter
from src.pipelineB.config import load_config

def resize_with_center_crop(image, target_size):
    """
    Resize image by scaling down to fit target size while preserving aspect ratio,
    then center crop to the target size.
    
    Args:
        image (numpy.ndarray): Input image (H, W, C) or (H, W) for depth maps
        target_size (tuple): Target size (height, width)
    
    Returns:
        numpy.ndarray: Resized and cropped image
    """
    target_height, target_width = target_size
    
    # Get current dimensions
    if len(image.shape) == 3:
        # RGB image
        h, w, _ = image.shape
    else:
        # Depth map
        h, w = image.shape
    
    # Calculate scaling factor to make the smaller dimension exactly target size
    scale = min(target_height / h, target_width / w)
    
    # Calculate new dimensions
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize while preserving aspect ratio
    if len(image.shape) == 3:
        # RGB image (interpolation=INTER_LINEAR)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        # Depth map (interpolation=INTER_NEAREST to preserve depth values)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # Center crop to target size
    # Calculate crop offsets
    start_y = max(0, (new_h - target_height) // 2)
    start_x = max(0, (new_w - target_width) // 2)
    
    # Perform the crop
    if len(resized.shape) == 3:
        # RGB image
        cropped = resized[start_y:start_y + target_height, start_x:start_x + target_width, :]
    else:
        # Depth map
        cropped = resized[start_y:start_y + target_height, start_x:start_x + target_width]
    
    # Handle edge cases where the resized image is smaller than the target
    # (this shouldn't happen with our approach but just in case)
    if cropped.shape[0] < target_height or cropped.shape[1] < target_width:
        # Create target sized image
        if len(image.shape) == 3:
            # RGB image
            result = np.zeros((target_height, target_width, image.shape[2]), dtype=image.dtype)
        else:
            # Depth map
            result = np.zeros((target_height, target_width), dtype=image.dtype)
        
        # Calculate paste position
        paste_y = (target_height - cropped.shape[0]) // 2
        paste_x = (target_width - cropped.shape[1]) // 2
        
        # Paste the image
        if len(image.shape) == 3:
            # RGB image
            result[paste_y:paste_y + cropped.shape[0], paste_x:paste_x + cropped.shape[1], :] = cropped
        else:
            # Depth map
            result[paste_y:paste_y + cropped.shape[0], paste_x:paste_x + cropped.shape[1]] = cropped
        
        return result
    
    return cropped

class RGBClassificationDataset(Sun3DBaseDataset):
    """
    Dataset for RGB-based classification using monocular depth estimation.
    """
    def __init__(self, data_root, sequences, split='train', transform=None, image_size=384):
        """
        Initialize the dataset.
        
        Args:
            data_root (str): Root directory of the dataset
            sequences (list): List of sequences to include
            split (str): 'train' or 'test'
            transform (callable, optional): Optional transform to be applied on a sample
            image_size (int): Target image size
        """
        super().__init__(data_root, sequences, split)
        self.transform = transform
        self.image_size = image_size
        
        # Define normalization for pre-trained models
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __getitem__(self, idx):
        """
        Get a data sample.
        
        Args:
            idx (int): Index
        
        Returns:
            dict: A data sample including RGB image, depth map, and labels
        """
        # Get base sample from Sun3DBaseDataset
        sample = super().__getitem__(idx)
        
        # Get RGB image
        rgb_image = sample['rgb_image']
        
        # Resize image using center crop to the required size
        rgb_image = resize_with_center_crop(rgb_image, (self.image_size, self.image_size))
        
        # Convert to tensor
        rgb_tensor = torch.from_numpy(rgb_image.transpose(2, 0, 1).astype(np.float32) / 255.0)
        
        # Apply normalization
        rgb_tensor = self.normalize(rgb_tensor)
        
        # Apply additional transforms if provided
        if self.transform is not None:
            rgb_tensor = self.transform(rgb_tensor)
        
        # Get label
        label = torch.tensor(sample['binary_label'], dtype=torch.long)
        
        # Get ground truth depth map if available
        gt_depth = None
        if 'depth_map' in sample and sample['depth_map'] is not None:
            # The depth map is already converted to meters in the base class
            depth_map = sample['depth_map'].astype(np.float32)
            
            # Resize depth using the same center crop approach
            if depth_map is not None:
                gt_depth = resize_with_center_crop(depth_map, (self.image_size, self.image_size))
                
                # Convert to tensor
                gt_depth = torch.from_numpy(gt_depth.astype(np.float32))
                
                # If depth is 3D (H,W,1), convert to 2D (H,W)
                if gt_depth.dim() == 3 and gt_depth.shape[2] == 1:
                    gt_depth = gt_depth.squeeze(2)
        
        # Create the final sample dictionary
        rgb_sample = {
            'rgb_image': rgb_tensor,
            'gt_depth': gt_depth,
            'label': label,
            'sequence': sample['sequence'],
            'subdir': sample['subdir'],
            'file_path': os.path.basename(self.data_pairs[idx]['rgb_file'])
        }
        
        return rgb_sample
    
    def __len__(self):
        """
        Get the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.data_pairs)

class RGBTransform:
    """
    Transformations for RGB images.
    """
    def __init__(self, mode='train', image_size=224):
        """
        Initialize RGB transforms.
        
        Args:
            mode (str): 'train', 'val', or 'test'
            image_size (int): Size to resize images to
        """
        self.mode = mode
        self.image_size = image_size
        
        # Define transforms
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                # We don't need RandomCrop since we already resize in the dataset
            ])
        else:
            # No data augmentation for validation and testing
            self.transform = None
    
    def __call__(self, image):
        """
        Apply transforms to an image.
        
        Args:
            image (torch.Tensor): Input image
        
        Returns:
            torch.Tensor: Transformed image
        """
        if self.transform is not None:
            # Convert to PIL Image
            if isinstance(image, torch.Tensor):
                # If image is a tensor, convert to PIL
                if image.dim() == 3:  # (C, H, W)
                    image = transforms.ToPILImage()(image)
                    image = self.transform(image)
                    # Convert back to tensor
                    return transforms.ToTensor()(image)
            else:
                # If image is a numpy array
                image = Image.fromarray(image.astype(np.uint8))
                image = self.transform(image)
                return np.array(image)
        
        return image

def get_dataloaders(config_file):
    """
    Get train, validation, and test dataloaders.
    
    Args:
        config_file (str): Path to the configuration file
    
    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader)
    """
    # Load configuration
    config = load_config(config_file)
    
    # Get transforms
    train_transform = get_transform(config, 'train')
    val_transform = get_transform(config, 'val')
    test_transform = get_transform(config, 'test')
    
    # Get dataloaders
    train_dataloader = get_dataloader(config, 'train', train_transform, batch_size=config['data'].get('train_batch_size', config['data'].get('batch_size', 32)))
    val_dataloader = get_dataloader(config, 'val', val_transform, batch_size=config['data'].get('eval_batch_size', config['data'].get('batch_size', 4)))
    test_dataloader = get_dataloader(config, 'test', test_transform, batch_size=config['data'].get('eval_batch_size', config['data'].get('batch_size', 4)))
    
    return train_dataloader, val_dataloader, test_dataloader

def get_dataloader(config, split='train', transform=None, batch_size=None):
    """
    Get dataloader for Pipeline B.
    
    Args:
        config (dict): Configuration dictionary
        split (str): 'train', 'val', or 'test'
        transform (callable, optional): Transform to apply to the data
        batch_size (int, optional): Batch size to use, overriding config if provided
    
    Returns:
        torch.utils.data.DataLoader: DataLoader for the specified split
    """
    # If batch_size not provided, use the one from config
    if batch_size is None:
        # Use appropriate batch size based on mode (train vs eval)
        if split == 'train':
            batch_size = config['data'].get('train_batch_size', config['data'].get('batch_size', 32))
        else:
            batch_size = config['data'].get('eval_batch_size', config['data'].get('batch_size', 4))
    
    # Get train and test sequences
    train_sequences, test_sequences = DatasetSplitter.get_train_test_sequences()
    
    # Determine which sequences to use based on the split
    if split == 'test':
        sequences = test_sequences
    else:
        sequences = train_sequences
    
    # Create dataset
    dataset = RGBClassificationDataset(
        data_root=config['data']['root'],
        sequences=sequences,
        split=split,
        transform=transform,
        image_size=config['data']['image_size']
    )
    
    # If split is 'train' or 'val', split the training data
    if split in ['train', 'val']:
        # Calculate the number of samples for training and validation
        train_val_split = config['data']['train_val_split']
        num_samples = len(dataset)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        split_idx = int(num_samples * train_val_split)
        
        if split == 'train':
            subset_indices = indices[:split_idx]
        else:  # split == 'val'
            subset_indices = indices[split_idx:]
        
        # Create subset
        dataset = torch.utils.data.Subset(dataset, subset_indices)
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader

def get_transform(config, split='train'):
    """
    Get transform for Pipeline B.
    
    Args:
        config (dict): Configuration dictionary
        split (str): 'train', 'val', or 'test'
    
    Returns:
        callable: Transform function
    """
    # Use the RGBTransform class based on the split
    transform = RGBTransform(
        mode=split,
        image_size=config['data']['image_size']
    )
    
    return transform 