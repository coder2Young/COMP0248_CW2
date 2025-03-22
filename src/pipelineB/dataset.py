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

class RGBClassificationDataset(Sun3DBaseDataset):
    """
    Dataset class for RGB-based classification with depth estimation.
    """
    def __init__(self, data_root, sequences, split='train', transform=None, image_size=224):
        """
        Initialize the RGB classification dataset.
        
        Args:
            data_root (str): Root directory of the dataset
            sequences (list): List of sequences to include
            split (str): 'train', 'val', or 'test'
            transform (callable, optional): Optional transform to be applied on RGB images
            image_size (int): Size to resize images to
        """
        super().__init__(data_root, sequences, split, transform)
        self.image_size = image_size
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
            dict: A data sample including RGB image and labels
        """
        # Get base sample from Sun3DBaseDataset
        sample = super().__getitem__(idx)
        
        # Get RGB image
        rgb_image = sample['rgb_image']
        
        # Resize image to the required size
        rgb_image = cv2.resize(rgb_image, (self.image_size, self.image_size))
        
        # Convert to tensor
        rgb_tensor = torch.from_numpy(rgb_image.transpose(2, 0, 1).astype(np.float32) / 255.0)
        
        # Apply normalization
        rgb_tensor = self.normalize(rgb_tensor)
        
        # Apply additional transforms if provided
        if self.transform is not None:
            rgb_tensor = self.transform(rgb_tensor)
        
        # Get label
        label = torch.tensor(sample['binary_label'], dtype=torch.long)
        
        # Create the final sample dictionary
        rgb_sample = {
            'rgb_image': rgb_tensor,
            'label': label,
            'sequence': sample['sequence'],
            'subdir': sample['subdir'],
            'file_path': os.path.basename(self.data_pairs[idx]['rgb_file'])
        }
        
        return rgb_sample

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

def get_dataloader(config, split='train', transform=None):
    """
    Get dataloader for Pipeline B.
    
    Args:
        config (dict): Configuration dictionary
        split (str): 'train', 'val', or 'test'
        transform (callable, optional): Transform to apply to the data
    
    Returns:
        torch.utils.data.DataLoader: DataLoader for the specified split
    """
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
        dataset = data.Subset(dataset, subset_indices)
    
    # Create dataloader
    dataloader = data.DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
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

def get_dataloaders(config_file):
    """
    Get all dataloaders for Pipeline B.
    
    Args:
        config_file (str): Path to the YAML configuration file
    
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
    train_dataloader = get_dataloader(config, 'train', train_transform)
    val_dataloader = get_dataloader(config, 'val', val_transform)
    test_dataloader = get_dataloader(config, 'test', test_transform)
    
    return train_dataloader, val_dataloader, test_dataloader 