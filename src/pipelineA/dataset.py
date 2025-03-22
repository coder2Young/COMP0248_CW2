import torch
import torch.utils.data as data
import numpy as np
import os
import yaml
from src.data_utils.dataset import TablePointCloudDataset, DatasetSplitter
from src.data_utils.transforms import PointCloudTransform

def load_config(config_file):
    """
    Load configuration from a YAML file.
    
    Args:
        config_file (str): Path to the YAML configuration file
    
    Returns:
        dict: Configuration dictionary
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure numeric values are properly typed
    # Convert any numeric values that might be strings to the appropriate numeric types
    if 'training' in config:
        if 'lr' in config['training'] and isinstance(config['training']['lr'], str):
            config['training']['lr'] = float(config['training']['lr'])
        if 'weight_decay' in config['training'] and isinstance(config['training']['weight_decay'], str):
            config['training']['weight_decay'] = float(config['training']['weight_decay'])
        if 'lr_decay' in config['training'] and isinstance(config['training']['lr_decay'], str):
            config['training']['lr_decay'] = float(config['training']['lr_decay'])
        if 'lr_decay_step' in config['training'] and isinstance(config['training']['lr_decay_step'], str):
            config['training']['lr_decay_step'] = int(config['training']['lr_decay_step'])
        if 'early_stopping' in config['training'] and isinstance(config['training']['early_stopping'], str):
            config['training']['early_stopping'] = int(config['training']['early_stopping'])
        if 'epochs' in config['training'] and isinstance(config['training']['epochs'], str):
            config['training']['epochs'] = int(config['training']['epochs'])
    
    # Ensure correct types for data configuration
    if 'data' in config:
        if 'num_points' in config['data'] and isinstance(config['data']['num_points'], str):
            config['data']['num_points'] = int(config['data']['num_points'])
        if 'batch_size' in config['data'] and isinstance(config['data']['batch_size'], str):
            config['data']['batch_size'] = int(config['data']['batch_size'])
        if 'num_workers' in config['data'] and isinstance(config['data']['num_workers'], str):
            config['data']['num_workers'] = int(config['data']['num_workers'])
        if 'train_val_split' in config['data'] and isinstance(config['data']['train_val_split'], str):
            config['data']['train_val_split'] = float(config['data']['train_val_split'])
        if 'use_height' in config['data'] and isinstance(config['data']['use_height'], str):
            config['data']['use_height'] = config['data']['use_height'].lower() == 'true'
    
    # Ensure correct types for model configuration
    if 'model' in config:
        if 'emb_dims' in config['model'] and isinstance(config['model']['emb_dims'], str):
            config['model']['emb_dims'] = int(config['model']['emb_dims'])
        if 'k' in config['model'] and isinstance(config['model']['k'], str):
            config['model']['k'] = int(config['model']['k'])
        if 'dropout' in config['model'] and isinstance(config['model']['dropout'], str):
            config['model']['dropout'] = float(config['model']['dropout'])
    
    return config

def get_dataloader(config, split='train', transform=None):
    """
    Get dataloader for Pipeline A.
    
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
    dataset = TablePointCloudDataset(
        data_root=config['data']['root'],
        sequences=sequences,
        split=split,
        mode='classification',
        num_points=config['data']['num_points'],
        transform=transform,
        use_height=config['data']['use_height']
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
    Get transform for Pipeline A.
    
    Args:
        config (dict): Configuration dictionary
        split (str): 'train', 'val', or 'test'
    
    Returns:
        callable: Transform function
    """
    # For training, apply data augmentation
    if split == 'train':
        transform = PointCloudTransform(
            normalize=True,
            rotate=True,
            jitter=True,
            scale=True,
            translate=True
        )
    else:
        # For validation and testing, only normalize the point cloud
        transform = PointCloudTransform(
            normalize=True,
            rotate=False,
            jitter=False,
            scale=False,
            translate=False
        )
    
    return transform

def get_dataloaders(config_file):
    """
    Get all dataloaders for Pipeline A.
    
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