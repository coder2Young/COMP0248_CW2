import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Model implementation based on:
# https://github.com/antao97/dgcnn.pytorch/blob/master/model.py
# Author: Antonio Barbalace

def knn(x, k):
    """
    K-Nearest Neighbors for point clouds.
    
    Args:
        x (torch.Tensor): Point cloud of shape (B, C, N)
        k (int): Number of nearest neighbors
    
    Returns:
        torch.Tensor: Indices of k-nearest neighbors of shape (B, N, k)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    """
    Get edge features for DGCNN.
    
    Args:
        x (torch.Tensor): Point cloud of shape (B, C, N)
        k (int): Number of nearest neighbors
        idx (torch.Tensor, optional): Indices of k-nearest neighbors
    
    Returns:
        torch.Tensor: Edge features of shape (B, 2*C, N, k)
    """
    batch_size, num_dims, num_points = x.size()
    
    if idx is None:
        idx = knn(x, k=k)  # (B, N, k)
    
    device = x.device
    
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    
    x = x.transpose(2, 1).contiguous()  # (B, N, C)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    
    return feature

class DGCNN(nn.Module):
    """
    Dynamic Graph CNN for point cloud classification.
    
    This implementation is based on the paper:
    "Dynamic Graph CNN for Learning on Point Clouds" by Wang et al.
    """
    def __init__(self, emb_dims=1024, input_dim=3, k=20, dropout=0.5, num_classes=2):
        """
        Initialize the DGCNN model.
        
        Args:
            emb_dims (int): Dimension of embeddings
            input_dim (int): Input dimension (3 for xyz coordinates, 4 if using height)
            k (int): Number of nearest neighbors
            dropout (float): Dropout rate
            num_classes (int): Number of output classes
        """
        super(DGCNN, self).__init__()
        
        self.emb_dims = emb_dims
        self.input_dim = input_dim
        self.k = k
        self.dropout = dropout
        self.num_classes = num_classes
        
        # Edge feature extraction
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim*2, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # MLP for classification
        self.linear1 = nn.Linear(emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        
        self.linear3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x (torch.Tensor): Point cloud of shape (B, N, C)
        
        Returns:
            torch.Tensor: Logits of shape (B, num_classes)
        """
        batch_size = x.size(0)
        
        # Transpose to (B, C, N)
        x = x.transpose(2, 1)
        
        # Edge feature extraction
        x1 = get_graph_feature(x, k=self.k)
        x1 = self.conv1(x1)
        x1 = x1.max(dim=-1, keepdim=False)[0]
        
        x2 = get_graph_feature(x1, k=self.k)
        x2 = self.conv2(x2)
        x2 = x2.max(dim=-1, keepdim=False)[0]
        
        x3 = get_graph_feature(x2, k=self.k)
        x3 = self.conv3(x3)
        x3 = x3.max(dim=-1, keepdim=False)[0]
        
        x4 = get_graph_feature(x3, k=self.k)
        x4 = self.conv4(x4)
        x4 = x4.max(dim=-1, keepdim=False)[0]
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        
        x = self.conv5(x)
        
        # Global feature vector
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        
        # MLP for classification
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        
        x = self.linear3(x)
        
        return x

def get_model(config):
    """
    Get a DGCNN model based on configuration.
    """
    # Calculate input dimension based on features used
    input_dim = 3  # XYZ coordinates are always used
    if config['data'].get('use_rgb', False):
        input_dim += 3  # Add RGB channels
    
    # Create model
    model = DGCNN(
        emb_dims=config['model']['emb_dims'],
        input_dim=input_dim,
        k=config['model']['k'],
        dropout=config['model']['dropout'],
        num_classes=2  # Binary classification: Table / No Table
    )
    
    return model 