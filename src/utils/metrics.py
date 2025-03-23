import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def compute_classification_metrics(y_true, y_pred):
    """
    Compute metrics for classification task.
    
    Args:
        y_true (numpy.ndarray): Ground truth labels
        y_pred (numpy.ndarray): Predicted labels
    
    Returns:
        dict: Dictionary containing classification metrics
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    # Extract values from confusion matrix
    tn, fp, fn, tp = conf_matrix.ravel()
    
    # Compute specificity (true negative rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }

def compute_segmentation_metrics(y_true, y_pred, num_classes=2):
    """
    Compute metrics for segmentation task.
    
    Args:
        y_true (numpy.ndarray): Ground truth labels
        y_pred (numpy.ndarray): Predicted labels
        num_classes (int): Number of classes
    
    Returns:
        dict: Dictionary containing segmentation metrics
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # Flatten the arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Check if we have data for both classes
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    unique_labels = np.unique(np.concatenate([unique_true, unique_pred]))
    actual_num_classes = len(unique_labels)
    
    # Compute per-class metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Compute precision, recall, and F1 with different averaging methods
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Compute per-class metrics (for binary segmentation)
    if num_classes == 2:
        # Get per-class scores (safely handling cases where only one class is present)
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Initialize with default values
        precision_table = 0.0
        recall_table = 0.0
        f1_table = 0.0
        
        # Check if class 1 (table) is present in the scores
        if 1 in unique_labels and len(per_class_precision) > 1:
            precision_table = per_class_precision[1]
            recall_table = per_class_recall[1]
            f1_table = per_class_f1[1]
    else:
        precision_table = np.nan
        recall_table = np.nan
        f1_table = np.nan
    
    # Compute IoU for each class
    # Always create confusion matrix with all classes (0 to num_classes-1)
    # This ensures we have the expected dimensions even when only one class is present
    conf_matrix = confusion_matrix(
        y_true, 
        y_pred, 
        labels=list(range(num_classes))
    )
    
    # Class names for better readability
    class_names = ["background", "table"] if num_classes == 2 else [f"class_{i}" for i in range(num_classes)]
    
    iou_list = []
    for cls in range(num_classes):
        intersection = conf_matrix[cls, cls]
        union = np.sum(conf_matrix[cls, :]) + np.sum(conf_matrix[:, cls]) - intersection
        iou = intersection / union if union > 0 else 0.0
        iou_list.append(iou)
    
    mean_iou = np.mean(iou_list)
    
    # Create metrics dictionary with standardized naming
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'precision_table': precision_table,
        'recall_table': recall_table,
        'f1_table': f1_table,
        'mean_iou': mean_iou,
    }
    
    # Add class-specific IoU values with descriptive names
    for cls in range(num_classes):
        metrics[f'iou_{class_names[cls]}'] = iou_list[cls]
    
    return metrics

def compute_metrics_from_logits(logits, targets, task='classification'):
    """
    Compute metrics from logits.
    
    Args:
        logits (torch.Tensor): Raw model outputs before softmax/sigmoid
        targets (torch.Tensor): Ground truth labels
        task (str): 'classification' or 'segmentation'
    
    Returns:
        dict: Dictionary containing metrics
    """
    # Convert logits to predictions
    if task == 'classification':
        preds = torch.argmax(logits, dim=1) if logits.shape[1] > 1 else (torch.sigmoid(logits) > 0.5).long().squeeze()
        
        # Compute classification metrics
        return compute_classification_metrics(targets, preds)
    else:  # task == 'segmentation'
        preds = torch.argmax(logits, dim=1) if logits.shape[1] > 1 else (torch.sigmoid(logits) > 0.5).long().squeeze()
        
        # Compute segmentation metrics
        num_classes = logits.shape[1] if logits.shape[1] > 1 else 2
        return compute_segmentation_metrics(targets, preds, num_classes=num_classes) 