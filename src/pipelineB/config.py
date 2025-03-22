import yaml
import os
import json

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
        if 'batch_size' in config['data'] and isinstance(config['data']['batch_size'], str):
            config['data']['batch_size'] = int(config['data']['batch_size'])
        if 'num_workers' in config['data'] and isinstance(config['data']['num_workers'], str):
            config['data']['num_workers'] = int(config['data']['num_workers'])
        if 'train_val_split' in config['data'] and isinstance(config['data']['train_val_split'], str):
            config['data']['train_val_split'] = float(config['data']['train_val_split'])
        if 'image_size' in config['data'] and isinstance(config['data']['image_size'], str):
            config['data']['image_size'] = int(config['data']['image_size'])
    
    # Ensure correct types for model configuration
    if 'model' in config:
        if 'pretrained' in config['model'] and isinstance(config['model']['pretrained'], str):
            config['model']['pretrained'] = config['model']['pretrained'].lower() == 'true'
        if 'depth_model_type' in config['model'] and isinstance(config['model']['depth_model_type'], str):
            # Keep as string, just make sure it's not None or empty
            if not config['model']['depth_model_type']:
                config['model']['depth_model_type'] = 'MiDaS_small'
    
    return config

def save_config(config, save_path):
    """
    Save configuration to a JSON file.
    
    Args:
        config (dict): Configuration dictionary
        save_path (str): Path to save the configuration
    """
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Configuration saved to {save_path}") 