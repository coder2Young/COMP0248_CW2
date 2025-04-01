import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaleInvariantLoss(nn.Module):
    """
    Scale-invariant MSE loss as described in Eigen et al.
    For monocular depth estimation, this loss is invariant to global scaling.
    """
    def __init__(self, alpha=0.5):
        """
        Initialize scale-invariant loss.
        
        Args:
            alpha (float): Weight for the variance term
        """
        super(ScaleInvariantLoss, self).__init__()
        self.alpha = alpha
    
    def forward(self, pred, target, mask=None):
        """
        Forward pass.
        
        Args:
            pred (torch.Tensor): Predicted depth maps
            target (torch.Tensor): Ground truth depth maps
            mask (torch.Tensor, optional): Mask for valid depth values
        
        Returns:
            torch.Tensor: Scale-invariant loss
        """
        # Apply mask if provided
        if mask is not None:
            pred = pred[mask]
            target = target[mask]
            
            # Return 0 if no valid pixels
            if pred.numel() == 0:
                return torch.tensor(0.0, device=pred.device)
        
        # Convert to log space
        pred_log = torch.log(pred + 1e-6)
        target_log = torch.log(target + 1e-6)
        
        # Compute log difference
        log_diff = pred_log - target_log
        
        # Compute terms
        term1 = torch.mean(log_diff ** 2)
        term2 = self.alpha * (torch.mean(log_diff) ** 2)
        
        return term1 - term2

class EdgeAwareGradientLoss(nn.Module):
    """
    Edge-aware gradient loss for monocular depth estimation.
    Encourages the model to preserve depth discontinuities.
    """
    def __init__(self):
        """
        Initialize edge-aware gradient loss.
        """
        super(EdgeAwareGradientLoss, self).__init__()
        # Define Sobel filters for gradient computation
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                   dtype=torch.float32).reshape(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                   dtype=torch.float32).reshape(1, 1, 3, 3)
    
    def forward(self, pred, target, mask=None):
        """
        Forward pass.
        
        Args:
            pred (torch.Tensor): Predicted depth maps (B, H, W)
            target (torch.Tensor): Ground truth depth maps (B, H, W)
            mask (torch.Tensor, optional): Mask for valid depth values (B, H, W)
        
        Returns:
            torch.Tensor: Edge-aware gradient loss
        """
        device = pred.device
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)
        
        # Add channel dimension if needed
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        # Create a valid mask
        if mask is None:
            mask = torch.ones_like(target)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        
        # Compute gradients
        pred_grad_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding=1)
        target_grad_x = F.conv2d(target, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1)
        
        # Compute gradients magnitude
        pred_grad = torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2 + 1e-6)
        target_grad = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2 + 1e-6)
        
        # Initialize valid_mask first, then check its shape
        valid_mask = mask
        
        # Apply mask - adjust to account for the convolution operation
        # Resize mask to match the gradient tensor size
        if valid_mask.shape[2:] != pred_grad.shape[2:]:
            # Convert valid mask to same size as gradient tensors
            valid_mask = mask[:, :, 1:-1, 1:-1]
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        # Compute loss (L1 on gradients)
        loss = torch.abs(pred_grad - target_grad) * valid_mask
        return loss.sum() / (valid_mask.sum() + 1e-6)

class DepthEstimationLoss(nn.Module):
    """
    Combined loss for depth estimation.
    """
    def __init__(self, si_weight=1.0, edge_weight=0.1):
        """
        Initialize combined depth estimation loss.
        
        Args:
            si_weight (float): Weight for scale-invariant loss
            edge_weight (float): Weight for edge-aware gradient loss
        """
        super(DepthEstimationLoss, self).__init__()
        self.si_loss = ScaleInvariantLoss()
        self.edge_loss = EdgeAwareGradientLoss()
        self.si_weight = si_weight
        self.edge_weight = edge_weight
    
    def forward(self, pred, target):
        """
        Forward pass.
        
        Args:
            pred (torch.Tensor): Predicted depth maps
            target (torch.Tensor): Ground truth depth maps
        
        Returns:
            tuple: (total_loss, loss_dict)
        """
        # Create a valid mask
        mask = (target > 0) & torch.isfinite(target) & torch.isfinite(pred)
        
        # Skip if no valid pixels
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device), {
                'si_loss': 0.0,
                'edge_loss': 0.0
            }
        
        # Compute individual losses
        si_loss_val = self.si_loss(pred, target, mask)
        edge_loss_val = self.edge_loss(pred, target, mask)
        
        # Compute total loss
        total_loss = self.si_weight * si_loss_val + self.edge_weight * edge_loss_val
        
        # Create loss dictionary for logging
        loss_dict = {
            'si_loss': si_loss_val.item(),
            'edge_loss': edge_loss_val.item()
        }
        
        return total_loss, loss_dict 