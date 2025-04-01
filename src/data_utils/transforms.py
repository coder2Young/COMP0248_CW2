import numpy as np
import torch
import torchvision.transforms as transforms
import random
import cv2

class RandomRotation:
    """
    Randomly rotate the point cloud around the Z axis.
    """
    def __init__(self, angle_range=(-180, 180)):
        self.angle_range = angle_range
    
    def __call__(self, point_cloud):
        angle = np.random.uniform(*self.angle_range) * np.pi / 180
        
        # Create rotation matrix
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        
        # Only rotate XYZ coordinates, preserve other channels
        xyz = point_cloud[:, :3]
        rotated_xyz = np.matmul(xyz, rotation_matrix.T)
        
        if point_cloud.shape[1] > 3:
            # If point cloud has more than 3 dimensions (e.g., RGB), keep those dimensions
            other_features = point_cloud[:, 3:]
            rotated_point_cloud = np.column_stack((rotated_xyz, other_features))
        else:
            rotated_point_cloud = rotated_xyz
        
        return rotated_point_cloud

class RandomScale:
    """
    Randomly scale the point cloud.
    """
    def __init__(self, scale_range=(0.8, 1.2)):
        self.scale_range = scale_range
    
    def __call__(self, point_cloud):
        scale = np.random.uniform(*self.scale_range)
        
        # Only scale XYZ coordinates, preserve other channels
        xyz = point_cloud[:, :3]
        scaled_xyz = xyz * scale
        
        if point_cloud.shape[1] > 3:
            # If point cloud has more than 3 dimensions (e.g., RGB), keep those dimensions
            other_features = point_cloud[:, 3:]
            scaled_point_cloud = np.column_stack((scaled_xyz, other_features))
        else:
            scaled_point_cloud = scaled_xyz
        
        return scaled_point_cloud

class RandomTranslation:
    """
    Randomly translate the point cloud.
    """
    def __init__(self, translation_range=(-0.2, 0.2)):
        self.translation_range = translation_range
    
    def __call__(self, point_cloud):
        translation = np.random.uniform(*self.translation_range, size=3)
        
        # Only translate XYZ coordinates, preserve other channels
        xyz = point_cloud[:, :3]
        translated_xyz = xyz + translation
        
        if point_cloud.shape[1] > 3:
            # If point cloud has more than 3 dimensions (e.g., RGB), keep those dimensions
            other_features = point_cloud[:, 3:]
            translated_point_cloud = np.column_stack((translated_xyz, other_features))
        else:
            translated_point_cloud = translated_xyz
        
        return translated_point_cloud

class RandomJitter:
    """
    Randomly jitter the point cloud.
    """
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip
    
    def __call__(self, point_cloud):
        # Only jitter XYZ coordinates, preserve other channels
        xyz = point_cloud[:, :3]
        jitter = np.clip(self.sigma * np.random.randn(*xyz.shape), -self.clip, self.clip)
        jittered_xyz = xyz + jitter
        
        if point_cloud.shape[1] > 3:
            # If point cloud has more than 3 dimensions (e.g., RGB), keep those dimensions
            other_features = point_cloud[:, 3:]
            jittered_point_cloud = np.column_stack((jittered_xyz, other_features))
        else:
            jittered_point_cloud = jittered_xyz
        
        return jittered_point_cloud

class PointCloudNormalization:
    """
    Normalize point cloud to have zero mean and unit standard deviation.
    """
    def __call__(self, point_cloud):
        """
        Normalize point cloud.
        
        Args:
            point_cloud (torch.Tensor or numpy.ndarray): Point cloud of shape (N, 3)
        
        Returns:
            torch.Tensor or numpy.ndarray: Normalized point cloud
        """
        if isinstance(point_cloud, torch.Tensor):
            original_type = 'torch'
            device = point_cloud.device
            point_cloud = point_cloud.cpu().numpy()
        else:
            original_type = 'numpy'
        
        # Compute centroid
        centroid = np.mean(point_cloud, axis=0)
        
        # Center the point cloud
        centered_point_cloud = point_cloud - centroid
        
        # Scale the point cloud
        scale = np.max(np.sqrt(np.sum(centered_point_cloud ** 2, axis=1)))
        if scale > 0:
            scaled_point_cloud = centered_point_cloud / scale
        else:
            scaled_point_cloud = centered_point_cloud
        
        # Convert back to original type
        if original_type == 'torch':
            scaled_point_cloud = torch.from_numpy(scaled_point_cloud).to(device)
        
        return scaled_point_cloud

class PointCloudTransform:
    """
    Combined transformation for point clouds.
    """
    def __init__(self, normalize=True, rotate=True, jitter=True, scale=True, translate=False):
        """
        Initialize the combined transform.
        
        Args:
            normalize (bool): Whether to normalize the point cloud
            rotate (bool): Whether to apply random rotation
            jitter (bool): Whether to apply random jitter
            scale (bool): Whether to apply random scaling
            translate (bool): Whether to apply random translation
        """
        self.transforms = []
        
        if normalize:
            self.transforms.append(PointCloudNormalization())
        
        if rotate:
            self.transforms.append(RandomRotation())
        
        if jitter:
            self.transforms.append(RandomJitter())
        
        if scale:
            self.transforms.append(RandomScale())
        
        if translate:
            self.transforms.append(RandomTranslation())
    
    def __call__(self, sample):
        """
        Apply transformations to the point cloud in the sample.
        
        Args:
            sample (dict): Dictionary containing 'point_cloud' key
        
        Returns:
            dict: Transformed sample
        """
        # Make a copy of the sample
        transformed_sample = sample.copy()
        
        # Check if point_cloud is in the sample
        if 'point_cloud' not in transformed_sample:
            print("Warning: 'point_cloud' key not found in sample. Skipping transformation.")
            return transformed_sample
        
        # Get the point cloud
        point_cloud = transformed_sample['point_cloud']
        
        # Apply all transforms
        for transform in self.transforms:
            point_cloud = transform(point_cloud)
        
        # Update the sample
        transformed_sample['point_cloud'] = point_cloud
        
        return transformed_sample

class ImageColorJitter:
    """
    Apply color jitter to RGB images.
    """
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        """
        Initialize the color jitter transform.
        
        Args:
            brightness (float): Brightness jitter range
            contrast (float): Contrast jitter range
            saturation (float): Saturation jitter range
            hue (float): Hue jitter range
        """
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
    
    def __call__(self, sample):
        """
        Apply color jitter to the RGB image in the sample.
        
        Args:
            sample (dict): Dictionary containing 'rgb_image' key
        
        Returns:
            dict: Transformed sample
        """
        # Make a copy of the sample
        transformed_sample = sample.copy()
        
        # Get the RGB image
        rgb_image = transformed_sample['rgb_image']
        
        # Convert numpy array to PIL Image if needed
        if isinstance(rgb_image, np.ndarray):
            from PIL import Image
            rgb_image = Image.fromarray(rgb_image.astype('uint8'))
        
        # Apply color jitter
        rgb_image = self.color_jitter(rgb_image)
        
        # Convert back to numpy array if needed
        if isinstance(rgb_image, Image.Image):
            rgb_image = np.array(rgb_image)
        
        # Update the sample
        transformed_sample['rgb_image'] = rgb_image
        
        return transformed_sample

class ImageFlip:
    """
    Apply random horizontal flip to images.
    """
    def __init__(self, p=0.5):
        """
        Initialize the flip transform.
        
        Args:
            p (float): Probability of applying the flip
        """
        self.p = p
    
    def __call__(self, sample):
        """
        Apply horizontal flip to the images in the sample.
        
        Args:
            sample (dict): Dictionary containing 'rgb_image', 'depth_map', and 'binary_mask' keys
        
        Returns:
            dict: Transformed sample
        """
        # Make a copy of the sample
        transformed_sample = sample.copy()
        
        # Randomly decide whether to flip
        if random.random() < self.p:
            # Flip RGB image if present
            if 'rgb_image' in transformed_sample:
                transformed_sample['rgb_image'] = cv2.flip(transformed_sample['rgb_image'], 1)
            
            # Flip depth map if present
            if 'depth_map' in transformed_sample:
                transformed_sample['depth_map'] = cv2.flip(transformed_sample['depth_map'], 1)
            
            # Flip binary mask if present
            if 'binary_mask' in transformed_sample:
                transformed_sample['binary_mask'] = cv2.flip(transformed_sample['binary_mask'], 1)
        
        return transformed_sample

class ImageTransform:
    """
    Combined transformation for images.
    """
    def __init__(self, color_jitter=True, flip=True):
        """
        Initialize the combined transform.
        
        Args:
            color_jitter (bool): Whether to apply color jitter
            flip (bool): Whether to apply random flip
        """
        self.transforms = []
        
        if color_jitter:
            self.transforms.append(ImageColorJitter())
        
        if flip:
            self.transforms.append(ImageFlip())
    
    def __call__(self, sample):
        """
        Apply transformations to the images in the sample.
        
        Args:
            sample (dict): Dictionary containing image data
        
        Returns:
            dict: Transformed sample
        """
        # Make a copy of the sample
        transformed_sample = sample.copy()
        
        # Apply all transforms
        for transform in self.transforms:
            transformed_sample = transform(transformed_sample)
        
        return transformed_sample 