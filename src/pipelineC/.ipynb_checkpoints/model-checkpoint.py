import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    """
    Find k nearest neighbors for each point in the point cloud.
    
    Args:
        x (torch.Tensor): Input point cloud of shape (B, C, N)
        k (int): Number of neighbors to find
    
    Returns:
        torch.Tensor: Indices of nearest neighbors of shape (B, N, k)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    # Find the k nearest neighbors
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    # print(idx.shape)
    return idx

def get_graph_feature(x, k=20, idx=None):
    """
    Construct edge features for each point with its k nearest neighbors.
    
    Args:
        x (torch.Tensor): Input point cloud of shape (B, C, N)
        k (int): Number of neighbors to use
        idx (torch.Tensor, optional): Precomputed nearest neighbor indices
    
    Returns:
        torch.Tensor: Edge features of shape (B, 2*C, N, k)
    """
    batch_size, num_dims, num_points = x.size()
    
    # If nearest neighbor indices are not provided, compute them
    if idx is None:
        idx = knn(x, k=k)
    
    # Convert to device
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    
    # Reshape to get features
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    # Concatenate the features from neighbors with the central point
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class DGCNN_Seg(nn.Module):
    """
    Dynamic Graph CNN for segmentation tasks.
    Modified from https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
    """
    def __init__(self, k=20, emb_dims=1024, dropout=0.5, num_classes=2, input_dim=3):
        """
        Initialize DGCNN segmentation model.
        
        Args:
            k (int): Number of nearest neighbors
            emb_dims (int): Embedding dimensions
            dropout (float): Dropout rate
            num_classes (int): Number of classes for segmentation
            input_dim (int): Input dimension of point features
        """
        super(DGCNN_Seg, self).__init__()
        
        # Parameters
        self.k = k
        self.emb_dims = emb_dims
        self.dropout = dropout
        self.num_classes = num_classes
        self.input_dim = input_dim
        
        # Edge convolution layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        
        # First conv layer input channels is 2*input_dim because get_graph_feature doubles the features
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2),
            # nn.Conv1d(64, 64, kernel_size=1, bias=False),
            # self.bn1,
            # nn.LeakyReLU(negative_slope=0.2),
            # nn.Conv1d(64, 64, kernel_size=1, bias=False),
            # self.bn1,
            # nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2),
            # nn.Conv2d(64, 64, kernel_size=1, bias=False),
            # self.bn2,
            # nn.LeakyReLU(negative_slope=0.2),
            # nn.Conv2d(64, 64, kernel_size=1, bias=False),
            # self.bn2,
            # nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2),
            # nn.Conv2d(64, 64, kernel_size=1, bias=False),
            # self.bn3,
            # nn.LeakyReLU(negative_slope=0.2),
            # nn.Conv2d(64, 64, kernel_size=1, bias=False),
            # self.bn3,
            # nn.LeakyReLU(negative_slope=0.2),
        )
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
        #     self.bn4,
        #     nn.LeakyReLU(negative_slope=0.2),
        # )
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
        #     self.bn5,
        #     nn.LeakyReLU(negative_slope=0.2),
        # )
        
        # Global feature embedding
        self.bn6 = nn.BatchNorm1d(emb_dims)
        self.conv6 = nn.Sequential(
            nn.Conv1d(64, emb_dims, kernel_size=1, bias=False),
            self.bn6,
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Per-point segmentation
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(128)
        
        self.conv7 = nn.Sequential(nn.Conv1d(emb_dims+64*3, 512, kernel_size=1, bias=False),
                                  self.bn7,
                                  nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                  self.bn8,
                                  nn.LeakyReLU(negative_slope=0.2))
        self.conv9 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                  self.bn9,
                                  nn.LeakyReLU(negative_slope=0.2))
        # self.dp1 = nn.Dropout(p=dropout)
        self.conv10 = nn.Conv1d(128, num_classes, kernel_size=1, bias=True)
        
    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input point cloud of shape (B, C, N)
        
        Returns:
            torch.Tensor: Per-point segmentation logits of shape (B, num_classes, N)
        """
        batch_size, num_dims, num_points = x.size()
        
        # Layer 1: Extract local features with edge convolution
        # x1 = get_graph_feature(x, k=self.k)
        x1 = self.conv1(x)
        # x1 = x1.max(dim=-1, keepdim=False)[0]
        
        # Layer 2
        x2 = get_graph_feature(x1, k=self.k)
        x2 = self.conv2(x2)
        x2 = x2.max(dim=-1, keepdim=False)[0]
        
        # Layer 3
        x3 = get_graph_feature(x2, k=self.k)
        x3 = self.conv3(x3)
        x3 = x3.max(dim=-1, keepdim=False)[0]
        
        # # Layer 4
        # x4 = get_graph_feature(x3, k=self.k)
        # x4 = self.conv4(x4)
        # x4 = x4.max(dim=-1, keepdim=False)[0]
        
        # # Layer 5
        # x5 = get_graph_feature(x4, k=self.k)
        # x5 = self.conv5(x5)
        # x5 = x5.max(dim=-1, keepdim=False)[0]
        
        # Concatenate all local features
        # x_cat = torch.cat((x1, x2, x3, x4, x5), dim=1)
        
        # Global features
        x6 = self.conv6(x3)
        x6 = x6.max(dim=-1, keepdim=True)[0]  # (B, emb_dims, 1)
        
        # Expand global features to all points
        x6 = x6.repeat(1, 1, num_points)
        
        # Concatenate global and local features
        x7 = torch.cat((x1, x2, x3, x6), dim=1)
        
        # Final segmentation layers
        x7 = self.conv7(x7)
        x7 = self.conv8(x7)
        # x7 = self.dp1(x7)
        x7 = self.conv9(x7)
        return self.conv10(x7)
        
class PointNet2_Seg(nn.Module):
    """
    PointNet++ for segmentation tasks.
    This is a simplified implementation focusing on the segmentation task.
    For a full implementation, one might want to use the official code or
    libraries like pytorch3d.
    """
    def __init__(self, num_classes=2, use_height=True):
        super(PointNet2_Seg, self).__init__()
        
        # This is a placeholder for a full PointNet++ implementation
        # In a real implementation, you would include set abstraction and
        # feature propagation layers as defined in the PointNet++ paper
        
        # For now, we'll just raise a NotImplementedError
        raise NotImplementedError("PointNet++ segmentation model is not implemented yet. Please use DGCNN instead.")

def get_model(config):
    """
    Get a model based on the provided configuration.
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        nn.Module: Model for Pipeline C
    """
    model_name = config['model']['name'].lower()
    
    # Calculate input dimension based on features
    input_dim = 3  # XYZ coordinates
    if config['data'].get('use_height', True):
        input_dim += 1  # Add height feature
    
    # Add RGB color channels (3)
    input_dim += 3
    
    if model_name == 'dgcnn':
        model = DGCNN_Seg(
            k=config['model']['k'],
            emb_dims=config['model']['emb_dims'],
            dropout=config['model']['dropout'],
            num_classes=config['model']['num_classes'],
            input_dim=input_dim
        )
    elif model_name == 'pointnet2':
        use_height = config['data'].get('use_height', True)
        model = PointNet2_Seg(
            num_classes=config['model']['num_classes'],
            use_height=use_height
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model 