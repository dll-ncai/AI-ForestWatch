# Copyright (c) 2023, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from sklearn.metrics import f1_score as f1
from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np


def precision(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    return precision_score(target.view(-1).cpu(), pred.view(-1).cpu())


def recall(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    return recall_score(target.view(-1).cpu(), pred.view(-1).cpu())


def f1_score(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    return f1(target.view(-1).cpu(), pred.view(-1).cpu())


def accuracy(output, target):
    with torch.no_grad():
        # Handle both list and tensor inputs
        if isinstance(output, list):
            # If output is a list, use the first element
            pred = torch.argmax(output[0], dim=1)
        else:
            # If output is a tensor, use it directly
            pred = torch.argmax(output, dim=1)
        
        # print(f"type of pred: {type(pred)}")
        # print(f"shape of pred: {pred.shape}")
        # print(f"type of target: {type(target)}")
        # print(f"length of target: {len(target)}")
        # print(f"shape of target: {target.shape}")
        # print(f"type of output: {type(output)}")
        # print(f"length of output: {len(output)}")
        if isinstance(output, list):
            print(f"shape of output[0]: {output[0].shape}")
        else:
            print(f"shape of output: {output.shape}")
        
        assert pred.shape[0] == target.shape[0]  # Check batch size matches
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / (target.shape[1]*target.shape[2]*target.shape[0])


def compute_iou(pred, target, num_classes=2):
    """Compute IoU for each class."""
    ious = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        
        ious.append(iou)
    
    return ious


def compute_class_accuracy(pred, target, num_classes=2):
    """Compute accuracy for each class."""
    accuracies = []
    
    for cls in range(num_classes):
        target_cls = (target == cls)
        if target_cls.sum() == 0:
            accuracies.append(1.0)  # No pixels of this class, perfect accuracy
        else:
            correct = (pred[target_cls] == cls).sum()
            total = target_cls.sum()
            accuracies.append(correct / total)
    
    return accuracies


def compute_metrics(eval_pred):
    """Compute comprehensive metrics for Hugging Face Trainer."""
    predictions, labels = eval_pred
    
    # Handle different input formats
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Convert to numpy if needed
    if hasattr(predictions, 'cpu'):
        predictions = predictions.cpu().numpy()
    if hasattr(labels, 'cpu'):
        labels = labels.cpu().numpy()
    
    # # Get predicted classes
    # pred_classes = np.argmax(predictions, axis=1)
    
    # Check if predictions are already argmax results or raw logits
    if len(predictions.shape) == 4:  # Raw logits: [batch, classes, height, width]
        pred_classes = np.argmax(predictions, axis=1)  # [batch, height, width]
    else:  # Already argmax results: [batch, height, width]
        pred_classes = predictions

    # Flatten for metric computation
    pred_flat = pred_classes.flatten()
    labels_flat = labels.flatten()
    
    # Basic metrics
    precision = precision_score(labels_flat, pred_flat, average='weighted', zero_division=0)
    recall = recall_score(labels_flat, pred_flat, average='weighted', zero_division=0)
    f1 = f1_score(labels_flat, pred_flat, average='weighted', zero_division=0)
    mean_accuracy = accuracy_score(labels_flat, pred_flat)
    
    # IoU metrics
    ious = compute_iou(pred_flat, labels_flat, num_classes=2)
    mean_iou = np.mean(ious)
    
    # Class-specific accuracies
    class_accuracies = compute_class_accuracy(pred_flat, labels_flat, num_classes=2)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_accuracy': mean_accuracy,
        'mean_iou': mean_iou,
        'background_iou': ious[0],
        'forest_iou': ious[1],
        'background_accuracy': class_accuracies[0],
        'forest_accuracy': class_accuracies[1]
    } # Copyright (c) 2023, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from sklearn.metrics import f1_score as f1
from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np


def precision(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    return precision_score(target.view(-1).cpu(), pred.view(-1).cpu())


def recall(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    return recall_score(target.view(-1).cpu(), pred.view(-1).cpu())


def f1_score(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    return f1(target.view(-1).cpu(), pred.view(-1).cpu())


def accuracy(output, target):
    with torch.no_grad():
        # Handle both list and tensor inputs
        if isinstance(output, list):
            # If output is a list, use the first element
            pred = torch.argmax(output[0], dim=1)
        else:
            # If output is a tensor, use it directly
            pred = torch.argmax(output, dim=1)
        
        # print(f"type of pred: {type(pred)}")
        # print(f"shape of pred: {pred.shape}")
        # print(f"type of target: {type(target)}")
        # print(f"length of target: {len(target)}")
        # print(f"shape of target: {target.shape}")
        # print(f"type of output: {type(output)}")
        # print(f"length of output: {len(output)}")
        if isinstance(output, list):
            print(f"shape of output[0]: {output[0].shape}")
        else:
            print(f"shape of output: {output.shape}")
        
        assert pred.shape[0] == target.shape[0]  # Check batch size matches
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / (target.shape[1]*target.shape[2]*target.shape[0])


def compute_iou(pred, target, num_classes=2):
    """Compute IoU for each class."""
    ious = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        
        ious.append(iou)
    
    return ious


def compute_class_accuracy(pred, target, num_classes=2):
    """Compute accuracy for each class."""
    accuracies = []
    
    for cls in range(num_classes):
        target_cls = (target == cls)
        if target_cls.sum() == 0:
            accuracies.append(1.0)  # No pixels of this class, perfect accuracy
        else:
            correct = (pred[target_cls] == cls).sum()
            total = target_cls.sum()
            accuracies.append(correct / total)
    
    return accuracies


def compute_metrics(eval_pred):
    """Compute comprehensive metrics for Hugging Face Trainer."""
    predictions, labels = eval_pred
    
    # Handle different input formats
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Convert to numpy if needed
    if hasattr(predictions, 'cpu'):
        predictions = predictions.cpu().numpy()
    if hasattr(labels, 'cpu'):
        labels = labels.cpu().numpy()
    
    # # Get predicted classes
    # pred_classes = np.argmax(predictions, axis=1)
    
    # Check if predictions are already argmax results or raw logits
    if len(predictions.shape) == 4:  # Raw logits: [batch, classes, height, width]
        pred_classes = np.argmax(predictions, axis=1)  # [batch, height, width]
    else:  # Already argmax results: [batch, height, width]
        pred_classes = predictions

    # Flatten for metric computation
    pred_flat = pred_classes.flatten()
    labels_flat = labels.flatten()
    
    # Basic metrics
    precision = precision_score(labels_flat, pred_flat, average='weighted', zero_division=0)
    recall = recall_score(labels_flat, pred_flat, average='weighted', zero_division=0)
    f1 = f1_score(labels_flat, pred_flat, average='weighted', zero_division=0)
    mean_accuracy = accuracy_score(labels_flat, pred_flat)
    
    # IoU metrics
    ious = compute_iou(pred_flat, labels_flat, num_classes=2)
    mean_iou = np.mean(ious)
    
    # Class-specific accuracies
    class_accuracies = compute_class_accuracy(pred_flat, labels_flat, num_classes=2)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_accuracy': mean_accuracy,
        'mean_iou': mean_iou,
        'background_iou': ious[0],
        'forest_iou': ious[1],
        'background_accuracy': class_accuracies[0],
        'forest_accuracy': class_accuracies[1]
    } 
