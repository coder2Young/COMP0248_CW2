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
    Convert depth map to point cloud using camera intrinsics.
    
    Args:
        depth_map (numpy.ndarray): Input depth map
        intrinsics (dict): Camera intrinsics including fx, fy, cx, cy
        subsample (bool): Whether to subsample the point cloud
        num_points (int): Number of points to sample if subsample is True
        min_depth (float): Minimum valid depth value
        max_depth (float): Maximum valid depth value
    
    Returns:
        numpy.ndarray: Point cloud of shape (N, 3) where N is num_points if subsample is True
                      or the number of valid depth pixels otherwise
    """
    # Check if depth map is valid
    if depth_map is None or depth_map.size == 0:
        print("Warning: Empty depth map received, returning zero point cloud")
        return np.zeros((num_points, 3))
    
    # Get depth dimensions
    height, width = depth_map.shape
    
    # Check intrinsics
    required_keys = ['fx', 'fy', 'cx', 'cy']
    if not all(key in intrinsics for key in required_keys):
        print(f"Warning: Missing intrinsics keys: {[key for key in required_keys if key not in intrinsics]}")
        # Use default values for missing keys
        default_values = {'fx': 525.0, 'fy': 525.0, 'cx': 319.5, 'cy': 239.5}
        for key in required_keys:
            if key not in intrinsics:
                intrinsics[key] = default_values[key]
    
    # Create pixel coordinates grid
    v, u = np.indices((height, width))
    
    # Get valid depth mask
    valid_mask = (depth_map > min_depth) & (depth_map < max_depth)
    
    # If no valid values, return zeros
    valid_count = np.sum(valid_mask)
    if valid_count == 0:
        print("Warning: No valid depth values found, returning zero point cloud")
        return np.zeros((num_points, 3))
    
    # Extract valid depth values and pixel coordinates
    z = depth_map.flatten() # [valid_mask]
    # print(depth_map.shape)
    # print(depth_map[valid_mask].shape)
    v_valid = v.flatten() # [valid_mask]
    u_valid = u.flatten() # [valid_mask]
    
    # Get intrinsics
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']
    
    # Calculate 3D coordinates
    x = (u_valid - cx) * z / fx
    y = (v_valid - cy) * z / fy
    
    # Stack the coordinates to form the point cloud
    point_cloud = np.stack((x, y, z), axis=-1)
    
    # Subsample the point cloud if needed
    if subsample and point_cloud.shape[0] > 0:
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
                padding = np.zeros((num_points - point_cloud.shape[0], 3))
            point_cloud = np.vstack((point_cloud, padding))
    
    # Ensure the output has the correct shape
    if point_cloud.shape[0] != num_points and subsample:
        print(f"Warning: Point cloud has {point_cloud.shape[0]} points, expected {num_points}")
        if point_cloud.shape[0] == 0:
            point_cloud = np.zeros((num_points, 3))
        else:
            # Resize by duplicating or truncating
            indices = np.random.choice(point_cloud.shape[0], num_points, replace=True)
            point_cloud = point_cloud[indices]
    
    return point_cloud

def read_and_parse_polygon_labels(labels_file, valid_table_labels=None):
    """
    Read and parse polygon annotations from a file.
    
    Args:
        labels_file (str): Path to the labels file
        valid_table_labels (list): List of valid table labels (not used for pickle format)
    
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
        
        # Try the legacy text parsing approach as fallback
        try:
            if valid_table_labels is None:
                valid_table_labels = ["table top", "dining table", "desk", "coffee table"]
            
            # Try different encodings and handle binary files
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']
            
            # Try to open as text with different encodings
            for encoding in encodings_to_try:
                try:
                    with open(labels_file, 'r', encoding=encoding) as f:
                        lines = f.readlines()
                        
                        current_timestamp = None
                        current_polygons = []
                        
                        for line in lines:
                            line = line.strip()
                            
                            # Skip empty lines
                            if not line:
                                continue
                            
                            # Check if line contains a timestamp
                            if line.startswith("timestamp:"):
                                # Save previous annotations if any
                                if current_timestamp is not None and current_polygons:
                                    annotations[current_timestamp] = current_polygons
                                
                                # Start new annotation
                                current_timestamp = line.split(":", 1)[1].strip()
                                current_polygons = []
                            
                            # Check if line contains a polygon
                            elif line.startswith("polygon:") and current_timestamp is not None:
                                # Extract label and points
                                polygon_info = line.split(":", 1)[1].strip()
                                
                                # Parse label and points (format may vary)
                                label_and_points = polygon_info.split(";")
                                
                                if len(label_and_points) >= 2:
                                    label = label_and_points[0].strip()
                                    
                                    # Only consider valid table labels
                                    if any(table_label.lower() in label.lower() for table_label in valid_table_labels):
                                        points_str = ";".join(label_and_points[1:])
                                        
                                        # Parse points (format may vary)
                                        points = []
                                        for point_str in points_str.split(","):
                                            try:
                                                x, y = map(float, point_str.strip().split())
                                                points.append((x, y))
                                            except:
                                                pass
                                        
                                        if points:
                                            current_polygons.append({
                                                "label": label,
                                                "points": points
                                            })
                        
                        # Save the last annotation if any
                        if current_timestamp is not None and current_polygons:
                            annotations[current_timestamp] = current_polygons
                        
                        # If we successfully parsed the file, return annotations
                        return annotations
                        
                except UnicodeDecodeError:
                    # Try the next encoding
                    continue
                except Exception as text_e:
                    # For other errors, try the next encoding
                    print(f"Error in text parsing fallback: {text_e}")
                    continue
            
            print(f"Warning: Could not parse file: {labels_file} with any method.")
            return {}
        except Exception as fallback_e:
            print(f"Error in fallback parsing for {labels_file}: {fallback_e}")
            return {}

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
    Convert depth map to colored point cloud using camera intrinsics and RGB image.
    
    Args:
        depth_map (numpy.ndarray): Input depth map
        rgb_image (numpy.ndarray): RGB image of shape (H, W, 3)
        intrinsics (dict): Camera intrinsics including fx, fy, cx, cy
        subsample (bool): Whether to subsample the point cloud
        num_points (int): Number of points to sample if subsample is True
        min_depth (float): Minimum valid depth value
        max_depth (float): Maximum valid depth value
    
    Returns:
        numpy.ndarray: Colored point cloud of shape (N, 6) where N is num_points if subsample is True
                      or the number of valid depth pixels otherwise. The 6 channels are XYZ coordinates
                      and RGB values normalized to [0, 1].
    """
    # First, generate XYZ point cloud using the existing function
    xyz_point_cloud = depth_to_point_cloud(
        depth_map, intrinsics, subsample=False, 
        min_depth=min_depth, max_depth=max_depth
    )
    
    # If no valid points were found, return zeros with RGB
    if xyz_point_cloud.shape[0] == 0:
        if subsample:
            return np.zeros((num_points, 6), dtype=np.float32)
        else:
            return np.zeros((0, 6), dtype=np.float32)
    
    # Get the original height and width
    height, width = depth_map.shape
    
    # Get camera intrinsics
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']
    
    # Initialize RGB values
    rgb_values = np.zeros((xyz_point_cloud.shape[0], 3), dtype=np.float32)
    
    # Project 3D points back to 2D image space to sample RGB
    for i, (x, y, z) in enumerate(xyz_point_cloud):
        if z <= 0:
            continue
            
        # Project to image coordinates
        u = int(x * fx / z + cx)
        v = int(y * fy / z + cy)
        
        # Check if projected point is inside the image
        if 0 <= u < width and 0 <= v < height:
            # Sample RGB value and normalize to [0, 1]
            rgb_values[i] = rgb_image[v, u] / 255.0
    
    # Combine XYZ and RGB to form 6-channel point cloud
    colored_point_cloud = np.column_stack((xyz_point_cloud, rgb_values))
    
    # Subsample if requested
    if subsample and colored_point_cloud.shape[0] > num_points:
        indices = np.random.choice(colored_point_cloud.shape[0], num_points, replace=False)
        colored_point_cloud = colored_point_cloud[indices]
    elif subsample and colored_point_cloud.shape[0] < num_points:
        # Pad with zeros if not enough points
        padding = np.zeros((num_points - colored_point_cloud.shape[0], 6), dtype=np.float32)
        colored_point_cloud = np.vstack([colored_point_cloud, padding])
    
    return colored_point_cloud.astype(np.float32) 