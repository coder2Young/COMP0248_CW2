import numpy as np
import torch
import torchvision.transforms as transforms
import random
import cv2

class PointCloudRotation:
    """
    Apply random rotation to point cloud.
    """
    def __init__(self, axis=None, angle_range=(-np.pi/4, np.pi/4)):
        """
        Initialize the rotation transform.
        
        Args:
            axis (int or None): Axis to rotate around (0, 1, 2 for x, y, z). If None, a random axis is chosen.
            angle_range (tuple): Range of angles to sample from
        """
        self.axis = axis
        self.angle_range = angle_range
    
    def __call__(self, point_cloud):
        """
        Apply rotation to point cloud.
        
        Args:
            point_cloud (torch.Tensor or numpy.ndarray): Point cloud of shape (N, 3)
        
        Returns:
            torch.Tensor or numpy.ndarray: Rotated point cloud
        """
        if isinstance(point_cloud, torch.Tensor):
            original_type = 'torch'
            device = point_cloud.device
            point_cloud = point_cloud.cpu().numpy()
        else:
            original_type = 'numpy'
        
        # Get random rotation axis if not specified
        if self.axis is None:
            axis = random.randint(0, 2)
        else:
            axis = self.axis
        
        # Get random angle
        angle = random.uniform(*self.angle_range)
        
        # Create rotation matrix
        rotation_matrix = np.eye(3)
        
        c, s = np.cos(angle), np.sin(angle)
        
        if axis == 0:  # Rotate around x-axis
            rotation_matrix[1, 1] = c
            rotation_matrix[1, 2] = -s
            rotation_matrix[2, 1] = s
            rotation_matrix[2, 2] = c
        elif axis == 1:  # Rotate around y-axis
            rotation_matrix[0, 0] = c
            rotation_matrix[0, 2] = s
            rotation_matrix[2, 0] = -s
            rotation_matrix[2, 2] = c
        else:  # Rotate around z-axis
            rotation_matrix[0, 0] = c
            rotation_matrix[0, 1] = -s
            rotation_matrix[1, 0] = s
            rotation_matrix[1, 1] = c
        
        # Apply rotation
        rotated_point_cloud = np.matmul(point_cloud, rotation_matrix.T)
        
        # Convert back to original type
        if original_type == 'torch':
            rotated_point_cloud = torch.from_numpy(rotated_point_cloud).to(device)
        
        return rotated_point_cloud

class PointCloudJitter:
    """
    Apply random jitter to point cloud.
    """
    def __init__(self, sigma=0.01, clip=0.05):
        """
        Initialize the jitter transform.
        
        Args:
            sigma (float): Standard deviation of the Gaussian noise
            clip (float): Maximum absolute value of the noise
        """
        self.sigma = sigma
        self.clip = clip
    
    def __call__(self, point_cloud):
        """
        Apply jitter to point cloud.
        
        Args:
            point_cloud (torch.Tensor or numpy.ndarray): Point cloud of shape (N, 3)
        
        Returns:
            torch.Tensor or numpy.ndarray: Jittered point cloud
        """
        if isinstance(point_cloud, torch.Tensor):
            original_type = 'torch'
            device = point_cloud.device
            point_cloud = point_cloud.cpu().numpy()
        else:
            original_type = 'numpy'
        
        # Generate noise
        noise = np.clip(self.sigma * np.random.randn(*point_cloud.shape), -self.clip, self.clip)
        
        # Apply noise
        jittered_point_cloud = point_cloud + noise
        
        # Convert back to original type
        if original_type == 'torch':
            jittered_point_cloud = torch.from_numpy(jittered_point_cloud).to(device)
        
        return jittered_point_cloud

class PointCloudScale:
    """
    Apply random scaling to point cloud.
    """
    def __init__(self, scale_range=(0.8, 1.2)):
        """
        Initialize the scaling transform.
        
        Args:
            scale_range (tuple): Range of scaling factors to sample from
        """
        self.scale_range = scale_range
    
    def __call__(self, point_cloud):
        """
        Apply scaling to point cloud.
        
        Args:
            point_cloud (torch.Tensor or numpy.ndarray): Point cloud of shape (N, 3)
        
        Returns:
            torch.Tensor or numpy.ndarray: Scaled point cloud
        """
        if isinstance(point_cloud, torch.Tensor):
            original_type = 'torch'
            device = point_cloud.device
            point_cloud = point_cloud.cpu().numpy()
        else:
            original_type = 'numpy'
        
        # Get random scaling factor
        scale = random.uniform(*self.scale_range)
        
        # Apply scaling
        scaled_point_cloud = point_cloud * scale
        
        # Convert back to original type
        if original_type == 'torch':
            scaled_point_cloud = torch.from_numpy(scaled_point_cloud).to(device)
        
        return scaled_point_cloud

class PointCloudTranslation:
    """
    Apply random translation to point cloud.
    """
    def __init__(self, translation_range=(-0.2, 0.2)):
        """
        Initialize the translation transform.
        
        Args:
            translation_range (tuple): Range of translation values to sample from
        """
        self.translation_range = translation_range
    
    def __call__(self, point_cloud):
        """
        Apply translation to point cloud.
        
        Args:
            point_cloud (torch.Tensor or numpy.ndarray): Point cloud of shape (N, 3)
        
        Returns:
            torch.Tensor or numpy.ndarray: Translated point cloud
        """
        if isinstance(point_cloud, torch.Tensor):
            original_type = 'torch'
            device = point_cloud.device
            point_cloud = point_cloud.cpu().numpy()
        else:
            original_type = 'numpy'
        
        # Get random translation for each axis
        translation = np.random.uniform(
            self.translation_range[0], 
            self.translation_range[1], 
            size=(1, 3)
        )
        
        # Apply translation
        translated_point_cloud = point_cloud + translation
        
        # Convert back to original type
        if original_type == 'torch':
            translated_point_cloud = torch.from_numpy(translated_point_cloud).to(device)
        
        return translated_point_cloud

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
            self.transforms.append(PointCloudRotation())
        
        if jitter:
            self.transforms.append(PointCloudJitter())
        
        if scale:
            self.transforms.append(PointCloudScale())
        
        if translate:
            self.transforms.append(PointCloudTranslation())
    
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