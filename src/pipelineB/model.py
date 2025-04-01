import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm

class DepthEstimationModel:
    """
    Wrapper class for MiDaS monocular depth estimation model.
    """
    def __init__(self, model_type="MiDaS_small"):
        """
        Initialize the depth estimation model.
        
        Args:
            model_type (str): Type of MiDaS model to use
        """
        self.model_type = model_type
        
        # Load model
        if model_type == "MiDaS_small":
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        elif model_type == "DPT_Large":
            self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        elif model_type == "DPT_Hybrid":
            self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Define transformation for input images
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    
    def to(self, device):
        """
        Move model to device.
        
        Args:
            device (torch.device): Device to move model to
        
        Returns:
            DepthEstimationModel: Self
        """
        self.model = self.model.to(device)
        return self
    
    @torch.no_grad()
    def predict(self, image):
        """
        Predict depth map for an image.
        
        Args:
            image (torch.Tensor or numpy.ndarray): RGB image
        
        Returns:
            torch.Tensor: Depth map
        """
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Apply transforms
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:  # Batch of images
                batch_size = image.size(0)
                depth_maps = []
                for i in range(batch_size):
                    transformed_image = self.transform(image[i].cpu().numpy().transpose(1, 2, 0)).to(image.device)
                    prediction = self.model(transformed_image)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=(image.size(2), image.size(3)),
                        mode="bicubic",
                        align_corners=False,
                    )
                    depth_maps.append(prediction.squeeze(1))
                return torch.stack(depth_maps)
            else:  # Single image
                transformed_image = self.transform(image.cpu().numpy().transpose(1, 2, 0)).to(image.device)
                prediction = self.model(transformed_image)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(image.size(1), image.size(2)),
                    mode="bicubic",
                    align_corners=False,
                )
                return prediction.squeeze(1)
        else:  # numpy array
            transformed_image = self.transform(image)
            if torch.cuda.is_available():
                transformed_image = transformed_image.cuda()
            prediction = self.model(transformed_image)
            return prediction.cpu().numpy()

class DepthClassifier(nn.Module):
    """
    CNN-based classifier for depth maps.
    Uses a modified ResNet18 as the base architecture.
    """
    def __init__(self, pretrained=True, num_classes=2):
        """
        Initialize the depth classifier.
        
        Args:
            pretrained (bool): Whether to use pretrained weights
            num_classes (int): Number of output classes
        """
        super(DepthClassifier, self).__init__()
        
        # Load ResNet18 model
        self.base_model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Modify the first conv layer to accept single-channel input (depth map)
        # self.base_model.conv1 = nn.Conv2d(
        #     1, 64, kernel_size=7, stride=2, padding=3, bias=False
        # )
        
        # Replace final fully connected layer
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input depth map of shape (B, 1, H, W)
        
        Returns:
            torch.Tensor: Logits of shape (B, num_classes)
        """
        return self.base_model(x)

class MonocularDepthClassifier(nn.Module):
    """
    End-to-end model for depth estimation and classification.
    """
    def __init__(self, depth_model_type="MiDaS_small", pretrained=True, num_classes=2, freeze_depth_estimator=True):
        """
        Initialize the monocular depth classifier.
        
        Args:
            depth_model_type (str): Type of MiDaS model to use
            pretrained (bool): Whether to use pretrained weights
            num_classes (int): Number of output classes
            freeze_depth_estimator (bool): Whether to freeze the depth estimator during training
        """
        super(MonocularDepthClassifier, self).__init__()
        
        # Initialize depth estimation model
        self.depth_estimator = DepthEstimationModel(model_type=depth_model_type)
        
        # Initialize depth classifier
        self.classifier = DepthClassifier(pretrained=pretrained, num_classes=num_classes)
        
        # Flag for training mode
        self.training_mode = False
        
        # Flag for freezing depth estimator
        self.freeze_depth_estimator = freeze_depth_estimator
    
    def to(self, device):
        """
        Move model to device.
        
        Args:
            device (torch.device): Device to move model to
        
        Returns:
            MonocularDepthClassifier: Self
        """
        self.depth_estimator.to(device)
        self.classifier = self.classifier.to(device)
        return self
    
    def train(self, mode=True):
        """
        Set the model to training mode.
        
        Args:
            mode (bool): Whether to set training mode
        
        Returns:
            MonocularDepthClassifier: Self
        """
        self.training_mode = mode
        # Always set classifier to training mode
        self.classifier.train(mode)
        
        # Set depth estimator mode based on freeze flag
        if self.freeze_depth_estimator:
            # If frozen, keep depth estimator in eval mode
            self.depth_estimator.model.eval()
        else:
            # If not frozen, allow depth estimator to train
            self.depth_estimator.model.train(mode)
        
        return self
    
    def eval(self):
        """
        Set the model to evaluation mode.
        
        Returns:
            MonocularDepthClassifier: Self
        """
        self.training_mode = False
        self.classifier.eval()
        self.depth_estimator.model.eval()
        return self
    
    def forward(self, x, return_depth=False):
        """
        Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input RGB image of shape (B, 3, H, W)
            return_depth (bool): Whether to return depth maps in addition to classification outputs
        
        Returns:
            torch.Tensor or tuple: Logits or (logits, depth_maps) if return_depth is True
        """
        # Conditionally use torch.no_grad() based on freeze_depth_estimator flag
        if self.freeze_depth_estimator:
            with torch.no_grad():
                depth_maps = self.depth_estimator.predict(x)
        else:
            depth_maps = self.depth_estimator.predict(x)
        
        # Normalize depth maps to [0, 1]
        batch_size = depth_maps.size(0) if depth_maps.dim() > 3 else 1
        if depth_maps.dim() <= 3:
            depth_maps = depth_maps.unsqueeze(0) if depth_maps.dim() == 2 else depth_maps.unsqueeze(1)
        
        # Store the original depth maps for return if needed
        original_depth_maps = depth_maps.clone()
        
        # Apply min-max normalization
        normalized_depth_maps = depth_maps.clone()
        for i in range(batch_size):
            depth_map = normalized_depth_maps[i] if batch_size > 1 else normalized_depth_maps
            min_val = torch.min(depth_map)
            max_val = torch.max(depth_map)
            if max_val > min_val:
                normalized_depth_maps[i] = (depth_map - min_val) / (max_val - min_val)
            else:
                normalized_depth_maps[i] = torch.zeros_like(depth_map)
        
        # Copy normalized depth maps on channel dimension to 3 channels
        depth_maps_3ch = normalized_depth_maps.repeat(1, 3, 1, 1)
        
        # Classify depth maps
        outputs = self.classifier(depth_maps_3ch)
        
        if return_depth:
            # Squeeze channel dimension if it exists to match GT depth shape (B, H, W)
            if original_depth_maps.dim() == 4 and original_depth_maps.size(1) == 1:
                original_depth_maps = original_depth_maps.squeeze(1)
            return outputs, original_depth_maps
        return outputs

def get_model(config):
    """
    Get a model based on the provided configuration.
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        MonocularDepthClassifier: Model for Pipeline B
    """
    model = MonocularDepthClassifier(
        depth_model_type=config['model']['depth_model_type'],
        pretrained=config['model']['pretrained'],
        num_classes=config['model']['num_classes'],
        freeze_depth_estimator=config['model'].get('freeze_depth_estimator', True)
    )
    
    return model 