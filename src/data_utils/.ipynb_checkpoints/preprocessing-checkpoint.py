import numpy as np
import cv2
import open3d as o3d
import os
import torch
import pickle

def read_intrinsics(intrinsics_file):
    """
    Read camera intrinsics from a file.
    
    Args:
        intrinsics_file (str): Path to the intrinsics file
    
    Returns:
        dict: Camera intrinsics including fx, fy, cx, cy
    """
    # Check if the file exists
    if not os.path.exists(intrinsics_file):
        raise FileNotFoundError(f"Intrinsics file not found: {intrinsics_file}")
    
    # Default values in case we can't parse the file
    default_intrinsics = {
        'fx': 525.0,
        'fy': 525.0,
        'cx': 319.5,
        'cy': 239.5
    }
    
    try:
        with open(intrinsics_file, 'r') as f:
            lines = f.readlines()
        
        # In Sun3D, intrinsics.txt contains a 3x3 camera matrix in the format:
        # fx 0 cx
        # 0 fy cy
        # 0 0 1
        if len(lines) >= 3:
            # Parse first row for fx and cx
            row1 = lines[0].strip().split()
            if len(row1) >= 3:
                fx = float(row1[0])
                cx = float(row1[2])
            else:
                print(f"Warning: Could not parse fx and cx from {intrinsics_file}. Using default values.")
                fx, cx = default_intrinsics['fx'], default_intrinsics['cx']
                
            # Parse second row for fy and cy
            row2 = lines[1].strip().split()
            if len(row2) >= 3:
                fy = float(row2[1])
                cy = float(row2[2])
            else:
                print(f"Warning: Could not parse fy and cy from {intrinsics_file}. Using default values.")
                fy, cy = default_intrinsics['fy'], default_intrinsics['cy']
                
            return {
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy
            }
        else:
            print(f"Warning: Intrinsics file {intrinsics_file} does not have enough lines. Using default values.")
            return default_intrinsics
            
    except Exception as e:
        print(f"Error reading intrinsics from {intrinsics_file}: {e}. Using default values.")
        return default_intrinsics

def normalize_depth(depth_map, min_depth=0.5, max_depth=10.0):
    """
    Normalize depth map to the range [0, 1].
    
    Args:
        depth_map (numpy.ndarray): Input depth map
        min_depth (float): Minimum valid depth value
        max_depth (float): Maximum valid depth value
    
    Returns:
        numpy.ndarray: Normalized depth map
    """
    # Convert to float if needed
    if depth_map.dtype != np.float32:
        depth_map = depth_map.astype(np.float32)
    
    # Get a mask of valid depth values
    valid_mask = (depth_map > min_depth) & (depth_map < max_depth)
    
    # If no valid values, return zeros
    if not np.any(valid_mask):
        return np.zeros_like(depth_map)
    
    # Normalize only valid values
    normalized_depth = np.zeros_like(depth_map)
    normalized_depth[valid_mask] = (depth_map[valid_mask] - min_depth) / (max_depth - min_depth)
    
    return normalized_depth

def depth_to_point_cloud(depth_map, intrinsics, subsample=True, num_points=1024, min_depth=0.5, max_depth=10.0):
    """
    Convert depth map to point cloud using camera intrinsics with Open3D.
    
    Args:
        depth_map (numpy.ndarray): Input depth map in meters
        intrinsics (dict): Camera intrinsics including fx, fy, cx, cy
        subsample (bool): Whether to subsample the point cloud
        num_points (int): Number of points to sample if subsample is True
        min_depth (float): Minimum valid depth value in meters
        max_depth (float): Maximum valid depth value in meters
    
    Returns:
        numpy.ndarray: Point cloud of shape (N, 3) where N is num_points if subsample is True
                      or the number of valid depth pixels otherwise
    """
    # Check if depth map is valid
    if depth_map is None or depth_map.size == 0:
        print("Warning: Empty depth map received, returning zero point cloud")
        return np.zeros((num_points, 3))
    
    # Create Open3D intrinsic object
    height, width = depth_map.shape
    
    # Ensure intrinsics are available
    required_keys = ['fx', 'fy', 'cx', 'cy']
    if not all(key in intrinsics for key in required_keys):
        print(f"Warning: Missing intrinsics keys: {[key for key in required_keys if key not in intrinsics]}")
        # Use default values for missing keys
        default_values = {'fx': 525.0, 'fy': 525.0, 'cx': 319.5, 'cy': 239.5}
        for key in required_keys:
            if key not in intrinsics:
                intrinsics[key] = default_values[key]
    
    # Create Open3D camera intrinsics
    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width, height, 
        intrinsics['fx'], intrinsics['fy'], 
        intrinsics['cx'], intrinsics['cy']
    )
    
    # Create depth image from numpy array
    # Open3D expects depth in meters, so we don't need to convert if already in meters
    depth_image = o3d.geometry.Image(depth_map.astype(np.float32))
    
    # Create point cloud from depth image
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_image, 
        o3d_intrinsics,
        depth_scale=1.0,  # depth is already in meters
        depth_trunc=max_depth,
        stride=1
    )
    
    # Get points as numpy array
    points = np.asarray(pcd.points)
    
    # Filter points based on min depth
    if points.shape[0] > 0:
        # Calculate depth of each point (z coordinate)
        depths = points[:, 2]
        valid_mask = depths >= min_depth
        points = points[valid_mask]
    
    # Check if we have any valid points
    if points.shape[0] == 0:
        print("Warning: No valid points after filtering, returning zero point cloud")
        return np.zeros((num_points, 3))
    
    # Subsample the point cloud if needed
    if subsample:
        if points.shape[0] > num_points:
            # Randomly sample points
            indices = np.random.choice(points.shape[0], num_points, replace=False)
            points = points[indices]
        elif points.shape[0] < num_points:
            # Pad with duplicated points or zeros if not enough points
            if points.shape[0] > 0:
                # Duplicate some points
                padding_indices = np.random.choice(points.shape[0], num_points - points.shape[0], replace=True)
                padding = points[padding_indices]
            else:
                # Use zeros if no valid points
                padding = np.zeros((num_points - points.shape[0], 3))
            points = np.vstack((points, padding))
    
    return points.astype(np.float32)

def read_and_parse_polygon_labels(labels_file):
    """
    Read and parse polygon annotations from a file.
    
    Args:
        labels_file (str): Path to the labels file
    
    Returns:
        dict: Dictionary mapping image timestamps to polygon annotations
    """
    # Check if the file exists
    if not os.path.exists(labels_file):
        print(f"Labels file not found: {labels_file}")
        return {}
    
    annotations = {}
    
    try:
        # Get the directory where labels_file is located
        labels_dir = os.path.dirname(labels_file)
        img_dir = os.path.join(os.path.dirname(labels_dir), 'image')
        
        # Load the table polygon labels
        with open(labels_file, 'rb') as label_file:
            tabletop_labels = pickle.load(label_file)
        
        # Get list of image files in the same order as the labels
        if os.path.exists(img_dir):
            img_list = sorted(os.listdir(img_dir))
            
            # Map each image to its corresponding polygon labels
            for i, (polygon_list, img_name) in enumerate(zip(tabletop_labels, img_list)):
                # Extract timestamp from image filename
                timestamp = img_name.split('.')[0]  # Remove file extension

                # Convert polygon format to our internal format
                # In pickle file, polygons are stored as [frame][table_instance][coordinate]
                # where coordinate is [x_coords, y_coords]
                formatted_polygons = []
                
                for polygon in polygon_list:
                    # Create a list of (x,y) tuples from the polygon coordinates
                    points = []
                    for x, y in zip(polygon[0], polygon[1]):
                        points.append((float(x), float(y)))
                    
                    if points:  # Only add if there are points
                        formatted_polygons.append({
                            "label": "table",  # Use generic label since pickle doesn't have table type
                            "points": points
                        })
                
                annotations[timestamp] = formatted_polygons
        else:
            print(f"Image directory not found: {img_dir}")
            
        return annotations
            
    except Exception as e:
        print(f"Error reading pickle annotations from {labels_file}: {e}")

def polygon_to_mask(polygon_points, image_shape):
    """
    Convert polygon points to binary mask.
    
    Args:
        polygon_points (list): List of (x, y) tuples representing polygon vertices
        image_shape (tuple): Shape of the image (height, width)
    
    Returns:
        numpy.ndarray: Binary mask of shape image_shape with 1s inside the polygon
    """
    # Create empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    # Convert polygon points to integer coordinates
    polygon_points = np.array(polygon_points, dtype=np.int32)
    
    # Draw filled polygon
    cv2.fillPoly(mask, [polygon_points], 1)
    
    return mask

def get_image_label_from_polygons(polygons, image_shape):
    """
    Generate image-level label based on polygon annotations.
    
    Args:
        polygons (list): List of dictionaries containing polygon annotations
        image_shape (tuple): Shape of the image (height, width)
    
    Returns:
        tuple: (binary_label, binary_mask) where binary_label is 1 if table is present, 0 otherwise
               and binary_mask is a mask with 1s inside the table polygons
    """
    # Create empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    # If no polygons, return 0 label and empty mask
    if not polygons:
        return 0, mask
    
    # Draw all polygons on the mask
    valid_polygon_count = 0
    for polygon in polygons:
        points = polygon["points"]
        if len(points) < 3:  # Need at least 3 points to form a polygon
            continue
            
        points_array = np.array(points, dtype=np.int32)
        
        # Check if points array is valid (not containing NaNs or out-of-bounds values)
        if np.any(np.isnan(points_array)) or np.any(points_array < 0) or \
           np.any(points_array[:, 0] >= image_shape[1]) or np.any(points_array[:, 1] >= image_shape[0]):
            # Skip invalid polygons
            continue
            
        cv2.fillPoly(mask, [points_array], 1)
        valid_polygon_count += 1
    
    # Binary label is 1 if there's at least one valid polygon (i.e., at least one table)
    binary_label = 1 if valid_polygon_count > 0 else 0
    
    return binary_label, mask

def point_cloud_to_depth(point_cloud, intrinsics, image_shape):
    """
    Convert point cloud back to depth map using camera intrinsics.
    This is useful for visualizing the point cloud.
    
    Args:
        point_cloud (numpy.ndarray): Point cloud of shape (N, 3)
        intrinsics (dict): Camera intrinsics including fx, fy, cx, cy
        image_shape (tuple): Shape of the output depth map (height, width)
    
    Returns:
        numpy.ndarray: Depth map of shape image_shape
    """
    height, width = image_shape
    depth_map = np.zeros(image_shape, dtype=np.float32)
    
    # Get intrinsics
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']
    
    # Project 3D points to 2D
    for x, y, z in point_cloud:
        if z <= 0:
            continue
        
        u = int(x * fx / z + cx)
        v = int(y * fy / z + cy)
        
        if 0 <= u < width and 0 <= v < height:
            depth_map[v, u] = z
    
    return depth_map

def depth_to_colored_point_cloud(depth_map, rgb_image, intrinsics, subsample=True, num_points=1024, min_depth=0.5, max_depth=10.0):
    """
    Convert depth map to colored point cloud using camera intrinsics and RGB image with Open3D.
    
    Args:
        depth_map (numpy.ndarray): Input depth map in meters
        rgb_image (numpy.ndarray): RGB image of shape (H, W, 3)
        intrinsics (dict): Camera intrinsics including fx, fy, cx, cy
        subsample (bool): Whether to subsample the point cloud
        num_points (int): Number of points to sample if subsample is True
        min_depth (float): Minimum valid depth value in meters
        max_depth (float): Maximum valid depth value in meters
    
    Returns:
        numpy.ndarray: Colored point cloud of shape (N, 6) where N is num_points if subsample is True
                      or the number of valid depth pixels otherwise. The 6 channels are XYZ coordinates
                      and RGB values normalized to [0, 1].
    """
    # Check if depth map and rgb image are valid
    if depth_map is None or depth_map.size == 0 or rgb_image is None or rgb_image.size == 0:
        print("Warning: Empty depth map or RGB image received, returning zero point cloud")
        return np.zeros((num_points, 6))
    
    # Create Open3D intrinsic object
    height, width = depth_map.shape
    
    # Ensure intrinsics are available
    required_keys = ['fx', 'fy', 'cx', 'cy']
    if not all(key in intrinsics for key in required_keys):
        print(f"Warning: Missing intrinsics keys: {[key for key in required_keys if key not in intrinsics]}")
        # Use default values for missing keys
        default_values = {'fx': 525.0, 'fy': 525.0, 'cx': 319.5, 'cy': 239.5}
        for key in required_keys:
            if key not in intrinsics:
                intrinsics[key] = default_values[key]
    
    # Create Open3D camera intrinsics
    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width, height, 
        intrinsics['fx'], intrinsics['fy'], 
        intrinsics['cx'], intrinsics['cy']
    )
    
    # Create depth image from numpy array
    depth_image = o3d.geometry.Image(depth_map.astype(np.float32))
    
    # Create RGB image (Open3D expects RGB images in uint8 format)
    rgb_image_o3d = o3d.geometry.Image(rgb_image.astype(np.uint8))
    
    # Create RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_image_o3d, 
        depth_image,
        depth_scale=1.0,  # depth is already in meters
        depth_trunc=max_depth,
        convert_rgb_to_intensity=False
    )
    
    # Create point cloud from RGBD image
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, 
        o3d_intrinsics
    )
    
    # Get points and colors as numpy arrays
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # Filter points based on min depth
    if points.shape[0] > 0:
        # Calculate depth of each point (z coordinate)
        depths = points[:, 2]
        valid_mask = depths >= min_depth
        points = points[valid_mask]
        colors = colors[valid_mask]
    
    # Check if we have any valid points
    if points.shape[0] == 0:
        print("Warning: No valid points after filtering, returning zero point cloud")
        return np.zeros((num_points, 6))
    
    # Combine points and colors into a single array
    point_cloud = np.column_stack((points, colors))
    
    # Subsample the point cloud if needed
    if subsample:
        if point_cloud.shape[0] > num_points:
            # Randomly sample points
            indices = np.random.choice(point_cloud.shape[0], num_points, replace=False)
            point_cloud = point_cloud[indices]
        elif point_cloud.shape[0] < num_points:
            # Pad with duplicated points or zeros if not enough points
            if point_cloud.shape[0] > 0:
                # Duplicate some points
                padding_indices = np.random.choice(point_cloud.shape[0], num_points - point_cloud.shape[0], replace=True)
                padding = point_cloud[padding_indices]
            else:
                # Use zeros if no valid points
                padding = np.zeros((num_points - point_cloud.shape[0], 6))
            point_cloud = np.vstack((point_cloud, padding))
    
    return point_cloud.astype(np.float32) 