"""Metrics calculation utilities for tracking evaluation."""

import numpy as np
from typing import Dict, List, Tuple, Optional


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
        
    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_distance(center1: Tuple[float, float], center2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        center1: (x, y) coordinates
        center2: (x, y) coordinates
        
    Returns:
        Euclidean distance
    """
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)


def get_box_center(box: List[float]) -> Tuple[float, float]:
    """
    Get center point of a bounding box.
    
    Args:
        box: [x1, y1, x2, y2] format
        
    Returns:
        (center_x, center_y)
    """
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def calculate_mota(gt_data: Dict, pred_data: Dict, iou_threshold: float = 0.5) -> Dict:
    """
    Calculate MOTA (Multiple Object Tracking Accuracy) and related metrics.
    
    Args:
        gt_data: Ground truth data {frame: [(x1, y1, x2, y2, track_id), ...]}
        pred_data: Prediction data {frame: [(x1, y1, x2, y2, track_id), ...]}
        iou_threshold: IoU threshold for matching
        
    Returns:
        Dictionary with MOTA metrics
    """
    total_gt = 0
    total_fp = 0
    total_fn = 0
    total_idsw = 0
    
    # Track ID mapping for ID switch detection
    id_mapping = {}
    
    for frame in sorted(set(gt_data.keys()) | set(pred_data.keys())):
        gt_boxes = gt_data.get(frame, [])
        pred_boxes = pred_data.get(frame, [])
        
        total_gt += len(gt_boxes)
        
        # Create matching matrix
        matches = []
        if gt_boxes and pred_boxes:
            iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
            
            for i, gt_box in enumerate(gt_boxes):
                for j, pred_box in enumerate(pred_boxes):
                    iou = calculate_iou(gt_box[:4], pred_box[:4])
                    iou_matrix[i, j] = iou
            
            # Find matches using Hungarian algorithm (simplified greedy approach)
            used_pred = set()
            for i in range(len(gt_boxes)):
                best_j = -1
                best_iou = 0
                for j in range(len(pred_boxes)):
                    if j not in used_pred and iou_matrix[i, j] > best_iou and iou_matrix[i, j] >= iou_threshold:
                        best_iou = iou_matrix[i, j]
                        best_j = j
                
                if best_j >= 0:
                    matches.append((i, best_j))
                    used_pred.add(best_j)
                    
                    # Check for ID switches
                    gt_id = gt_boxes[i][4]
                    pred_id = pred_boxes[best_j][4]
                    
                    if gt_id in id_mapping:
                        if id_mapping[gt_id] != pred_id:
                            total_idsw += 1
                            id_mapping[gt_id] = pred_id
                    else:
                        id_mapping[gt_id] = pred_id
        
        # Count false positives and false negatives
        total_fp += len(pred_boxes) - len(matches)
        total_fn += len(gt_boxes) - len(matches)
    
    # Calculate MOTA
    mota = 1 - (total_fp + total_fn + total_idsw) / total_gt if total_gt > 0 else 0
    
    return {
        'MOTA': mota,
        'total_gt': total_gt,
        'false_positives': total_fp,
        'false_negatives': total_fn,
        'id_switches': total_idsw,
        'precision': len(matches) / (len(matches) + total_fp) if (len(matches) + total_fp) > 0 else 0,
        'recall': len(matches) / total_gt if total_gt > 0 else 0
    }


def calculate_idf1(gt_data: Dict, pred_data: Dict, iou_threshold: float = 0.5) -> float:
    """
    Calculate IDF1 (Identity F1 Score).
    
    Args:
        gt_data: Ground truth data
        pred_data: Prediction data
        iou_threshold: IoU threshold for matching
        
    Returns:
        IDF1 score
    """
    # This is a simplified IDF1 calculation
    # For complete implementation, use specialized libraries like TrackEval
    
    total_idtp = 0  # Identity true positives
    total_idfp = 0  # Identity false positives
    total_idfn = 0  # Identity false negatives
    
    # Create association maps
    gt_associations = {}  # gt_id -> [(frame, box), ...]
    pred_associations = {}  # pred_id -> [(frame, box), ...]
    
    for frame in sorted(set(gt_data.keys()) | set(pred_data.keys())):
        gt_boxes = gt_data.get(frame, [])
        pred_boxes = pred_data.get(frame, [])
        
        for box in gt_boxes:
            gt_id = box[4]
            if gt_id not in gt_associations:
                gt_associations[gt_id] = []
            gt_associations[gt_id].append((frame, box[:4]))
        
        for box in pred_boxes:
            pred_id = box[4]
            if pred_id not in pred_associations:
                pred_associations[pred_id] = []
            pred_associations[pred_id].append((frame, box[:4]))
    
    # Calculate identity metrics (simplified)
    for gt_id, gt_track in gt_associations.items():
        # Find best matching predicted track
        best_overlap = 0
        best_pred_id = None
        
        for pred_id, pred_track in pred_associations.items():
            overlap = 0
            for gt_frame, gt_box in gt_track:
                for pred_frame, pred_box in pred_track:
                    if gt_frame == pred_frame:
                        if calculate_iou(gt_box, pred_box) >= iou_threshold:
                            overlap += 1
                        break
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_pred_id = pred_id
        
        if best_pred_id:
            total_idtp += best_overlap
            total_idfn += len(gt_track) - best_overlap
            total_idfp += len(pred_associations[best_pred_id]) - best_overlap
        else:
            total_idfn += len(gt_track)
    
    # Add unmatched predicted tracks
    matched_pred_ids = set()
    for gt_track in gt_associations.values():
        # This is simplified - in reality we'd need proper matching
        pass
    
    # Calculate IDF1
    idf1 = (2 * total_idtp) / (2 * total_idtp + total_idfp + total_idfn) if (2 * total_idtp + total_idfp + total_idfn) > 0 else 0
    
    return idf1


def calculate_track_quality_metrics(track_history: Dict) -> Dict:
    """
    Calculate track quality metrics.
    
    Args:
        track_history: Dictionary of track histories
        
    Returns:
        Dictionary with quality metrics
    """
    if not track_history:
        return {}
    
    track_lengths = [len(history) for history in track_history.values()]
    
    # Calculate fragmentation
    fragmentations = 0
    for history in track_history.values():
        frames = sorted([entry['frame'] for entry in history])
        gaps = 0
        for i in range(1, len(frames)):
            if frames[i] - frames[i-1] > 1:
                gaps += 1
        fragmentations += gaps
    
    return {
        'total_tracks': len(track_history),
        'avg_track_length': np.mean(track_lengths),
        'std_track_length': np.std(track_lengths),
        'min_track_length': min(track_lengths),
        'max_track_length': max(track_lengths),
        'track_fragmentations': fragmentations,
        'avg_fragmentation_per_track': fragmentations / len(track_history)
    }