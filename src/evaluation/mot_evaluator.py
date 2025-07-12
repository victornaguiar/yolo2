"""MOT evaluation metrics implementation."""

import os
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

try:
    import trackeval
    TRACKEVAL_AVAILABLE = True
except ImportError:
    print("Warning: TrackEval not available. Install with: pip install git+https://github.com/JonathonLuiten/TrackEval.git")
    TRACKEVAL_AVAILABLE = False

from .metrics import calculate_mota, calculate_idf1, calculate_track_quality_metrics


class MOTEvaluator:
    """Evaluator for Multi-Object Tracking results."""
    
    def __init__(self, 
                 metrics: List[str] = ['HOTA', 'CLEAR', 'Identity'],
                 threshold: float = 0.5):
        """
        Initialize MOT evaluator.
        
        Args:
            metrics: List of metrics to compute
            threshold: IoU threshold for matching
        """
        self.metrics = metrics
        self.threshold = threshold
        
        # Set logging level
        logging.getLogger().setLevel(logging.WARNING)
    
    def evaluate(self,
                 gt_dir: str,
                 results_dir: str,
                 sequences: Optional[List[str]] = None) -> Dict:
        """
        Evaluate tracking results against ground truth.
        
        Args:
            gt_dir: Directory containing ground truth files
            results_dir: Directory containing tracking results
            sequences: List of sequences to evaluate (None = all)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if TRACKEVAL_AVAILABLE:
            return self._evaluate_with_trackeval(gt_dir, results_dir, sequences)
        else:
            return self._evaluate_basic(gt_dir, results_dir, sequences)
    
    def _evaluate_with_trackeval(self,
                                gt_dir: str,
                                results_dir: str,
                                sequences: Optional[List[str]] = None) -> Dict:
        """Evaluate using TrackEval library."""
        gt_dir = Path(gt_dir)
        results_dir = Path(results_dir)
        
        # Get list of sequences to evaluate
        if sequences is None:
            sequences = self._get_sequences(results_dir)
        
        # Load data
        gt_data, tracker_data = self._load_data(gt_dir, results_dir, sequences)
        
        # Configure evaluation
        eval_config = {
            'USE_PARALLEL': False,
            'PRINT_ONLY_COMBINED': True
        }
        
        metrics_config = {
            'METRICS': self.metrics,
            'THRESHOLD': self.threshold
        }
        
        # Create dataset and evaluator
        dataset = InMemoryDataset(
            name='MOTDataset',
            tracker_name='Tracker',
            seq_list=sequences,
            gt_data=gt_data,
            tracker_data=tracker_data
        )
        
        evaluator = trackeval.Evaluator(eval_config)
        metrics_list = [
            getattr(trackeval.metrics, metric)(metrics_config) 
            for metric in self.metrics
        ]
        
        # Run evaluation
        results, _ = evaluator.evaluate([dataset], metrics_list)
        
        # Extract and format results
        return self._format_results(results, dataset.name, dataset.tracker_list[0])
    
    def _evaluate_basic(self,
                       gt_dir: str,
                       results_dir: str,
                       sequences: Optional[List[str]] = None) -> Dict:
        """Basic evaluation without TrackEval."""
        gt_dir = Path(gt_dir)
        results_dir = Path(results_dir)
        
        if sequences is None:
            sequences = self._get_sequences(results_dir)
        
        all_gt_data = {}
        all_pred_data = {}
        
        # Load and combine data from all sequences
        for seq in sequences:
            gt_file = gt_dir / f"{seq}.txt"
            results_file = results_dir / f"{seq}.txt"
            
            if gt_file.exists() and results_file.exists():
                gt_seq_data = self._load_mot_file(gt_file)
                pred_seq_data = self._load_mot_file(results_file)
                
                # Offset frame numbers by sequence to avoid conflicts
                frame_offset = len(all_gt_data)
                
                for frame, boxes in gt_seq_data.items():
                    all_gt_data[frame + frame_offset] = boxes
                
                for frame, boxes in pred_seq_data.items():
                    all_pred_data[frame + frame_offset] = boxes
        
        # Calculate metrics
        mota_results = calculate_mota(all_gt_data, all_pred_data, self.threshold)
        idf1_score = calculate_idf1(all_gt_data, all_pred_data, self.threshold)
        
        # Convert prediction data to track history format for quality metrics
        track_history = self._convert_to_track_history(all_pred_data)
        quality_metrics = calculate_track_quality_metrics(track_history)
        
        return {
            'MOTA': mota_results['MOTA'] * 100,
            'IDF1': idf1_score * 100,
            'precision': mota_results['precision'] * 100,
            'recall': mota_results['recall'] * 100,
            'false_positives': mota_results['false_positives'],
            'false_negatives': mota_results['false_negatives'],
            'id_switches': mota_results['id_switches'],
            'total_gt': mota_results['total_gt'],
            **quality_metrics
        }
    
    def _load_mot_file(self, file_path: Path) -> Dict[int, List]:
        """Load MOT format file into frame-based dictionary."""
        data = {}
        
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                
                frame_num = int(parts[0])
                track_id = int(parts[1])
                bb_left = float(parts[2])
                bb_top = float(parts[3])
                bb_width = float(parts[4])
                bb_height = float(parts[5])
                conf = float(parts[6]) if len(parts) > 6 else 1.0
                
                # Convert to [x1, y1, x2, y2, track_id] format
                box = [bb_left, bb_top, bb_left + bb_width, bb_top + bb_height, track_id]
                
                if frame_num not in data:
                    data[frame_num] = []
                data[frame_num].append(box)
        
        return data
    
    def _convert_to_track_history(self, pred_data: Dict) -> Dict:
        """Convert frame-based data to track history format."""
        track_history = {}
        
        for frame, boxes in pred_data.items():
            for box in boxes:
                track_id = int(box[4])
                if track_id not in track_history:
                    track_history[track_id] = []
                
                track_history[track_id].append({
                    'frame': frame,
                    'bbox': box[:4],
                    'confidence': 1.0  # Default confidence
                })
        
        return track_history
    
    def _get_sequences(self, results_dir: Path) -> List[str]:
        """Get list of sequences from results directory."""
        return sorted([
            f.stem for f in results_dir.iterdir() 
            if f.suffix == '.txt'
        ])
    
    def _load_data(self, 
                   gt_dir: Path, 
                   results_dir: Path, 
                   sequences: List[str]) -> Tuple[Dict, Dict]:
        """Load ground truth and tracking data for TrackEval."""
        gt_data = {}
        tracker_data = {}
        
        for seq in sequences:
            gt_file = gt_dir / f"{seq}.txt"
            results_file = results_dir / f"{seq}.txt"
            
            if gt_file.exists():
                gt_data[seq] = np.loadtxt(gt_file, delimiter=',', ndmin=2)
            else:
                print(f"Warning: GT file not found for {seq}")
                gt_data[seq] = np.array([]).reshape(0, 10)
            
            if results_file.exists():
                tracker_data[seq] = np.loadtxt(results_file, delimiter=',', ndmin=2)
            else:
                print(f"Warning: Results file not found for {seq}")
                tracker_data[seq] = np.array([]).reshape(0, 10)
        
        return gt_data, tracker_data
    
    def _format_results(self, 
                       results: Dict, 
                       dataset_name: str, 
                       tracker_name: str) -> Dict:
        """Format evaluation results."""
        formatted = {}
        
        dataset_results = results[dataset_name]
        tracker_results = dataset_results[tracker_name]['COMBINED_SEQ']['pedestrian']
        
        # Extract key metrics
        if 'HOTA' in tracker_results:
            formatted['HOTA'] = tracker_results['HOTA']['HOTA'] * 100
            formatted['DetA'] = tracker_results['HOTA']['DetA'] * 100
            formatted['AssA'] = tracker_results['HOTA']['AssA'] * 100
        
        if 'CLEAR' in tracker_results:
            formatted['MOTA'] = tracker_results['CLEAR']['MOTA'] * 100
            formatted['MOTP'] = tracker_results['CLEAR']['MOTP'] * 100
        
        if 'Identity' in tracker_results:
            formatted['IDF1'] = tracker_results['Identity']['IDF1'] * 100
        
        return formatted
    
    def print_results(self, metrics: Dict):
        """Print evaluation results in a formatted table."""
        print("\n" + "="*50)
        print("TRACKING EVALUATION RESULTS")
        print("="*50)
        
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                if metric in ['HOTA', 'MOTA', 'MOTP', 'IDF1', 'DetA', 'AssA', 'precision', 'recall']:
                    print(f"{metric:<15}: {value:>6.2f}%")
                else:
                    print(f"{metric:<15}: {value:>6}")
            else:
                print(f"{metric:<15}: {value}")
        
        print("="*50)


class InMemoryDataset:
    """In-memory dataset for TrackEval."""
    
    def __init__(self, name, tracker_name, seq_list, gt_data, tracker_data):
        self.name = name
        self.tracker_list = [tracker_name]
        self.seq_list = seq_list
        self.class_list = ['pedestrian']
        self.do_preproc = True
        self.gt_data = gt_data
        self.tracker_data = tracker_data
    
    def get_name(self):
        return self.name
    
    def get_eval_info(self):
        return self.tracker_list, self.seq_list, self.class_list
    
    def get_raw_seq_data(self, tracker, seq):
        return self.gt_data.get(seq, np.array([])), self.tracker_data.get(seq, np.array([]))
    
    def get_preprocessed_seq_data(self, raw_data, cls):
        """Preprocess data for evaluation."""
        gt_data_raw, track_data_raw = raw_data
        
        # Handle empty data
        if gt_data_raw.size == 0:
            gt_data_raw = np.array([]).reshape(0, 10)
        if track_data_raw.size == 0:
            track_data_raw = np.array([]).reshape(0, 10)
        
        # Get number of timesteps
        num_timesteps = 0
        if gt_data_raw.shape[0] > 0:
            num_timesteps = int(gt_data_raw[:, 0].max())
        if track_data_raw.shape[0] > 0:
            num_timesteps = max(num_timesteps, int(track_data_raw[:, 0].max()))
        
        if num_timesteps == 0:
            num_timesteps = 1
        
        # Initialize data structure
        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'similarity_scores']
        data = {key: [None] * num_timesteps for key in data_keys}
        
        # Process each timestep
        for t in range(num_timesteps):
            time_key = t + 1
            
            # Get GT data for this frame
            gt_in_frame = gt_data_raw[gt_data_raw[:, 0] == time_key] if gt_data_raw.shape[0] > 0 else np.array([])
            data['gt_ids'][t] = gt_in_frame[:, 1].astype(int) if gt_in_frame.shape[0] > 0 else np.array([])
            
            # Convert from [x, y, w, h] to [x1, y1, x2, y2]
            if gt_in_frame.shape[0] > 0:
                gt_boxes = gt_in_frame[:, 2:6].copy()
                gt_boxes[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2]  # x2 = x1 + width
                gt_boxes[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3]  # y2 = y1 + height
                data['gt_dets'][t] = gt_boxes
            else:
                data['gt_dets'][t] = np.array([]).reshape(0, 4)
            
            # Get tracker data for this frame
            tracker_in_frame = track_data_raw[track_data_raw[:, 0] == time_key] if track_data_raw.shape[0] > 0 else np.array([])
            data['tracker_ids'][t] = tracker_in_frame[:, 1].astype(int) if tracker_in_frame.shape[0] > 0 else np.array([])
            
            # Convert from [x, y, w, h] to [x1, y1, x2, y2]
            if tracker_in_frame.shape[0] > 0:
                tracker_boxes = tracker_in_frame[:, 2:6].copy()
                tracker_boxes[:, 2] = tracker_boxes[:, 0] + tracker_boxes[:, 2]  # x2 = x1 + width
                tracker_boxes[:, 3] = tracker_boxes[:, 1] + tracker_boxes[:, 3]  # y2 = y1 + height
                data['tracker_dets'][t] = tracker_boxes
            else:
                data['tracker_dets'][t] = np.array([]).reshape(0, 4)
            
            # Calculate similarities
            if data['gt_dets'][t].shape[0] > 0 and data['tracker_dets'][t].shape[0] > 0:
                data['similarity_scores'][t] = self._calculate_box_ious(
                    data['gt_dets'][t], 
                    data['tracker_dets'][t]
                )
            else:
                data['similarity_scores'][t] = np.array([]).reshape(0, 0)
        
        return data
    
    def _calculate_box_ious(self, bboxes1, bboxes2):
        """Calculate IoU between two sets of boxes."""
        ious = np.zeros((len(bboxes1), len(bboxes2)))
        
        for i, box1 in enumerate(bboxes1):
            for j, box2 in enumerate(bboxes2):
                # Calculate intersection
                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[2], box2[2])
                y2 = min(box1[3], box2[3])
                
                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                else:
                    intersection = 0
                
                # Calculate union
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                union = area1 + area2 - intersection
                
                # Calculate IoU
                if union > 0:
                    ious[i, j] = intersection / union
                else:
                    ious[i, j] = 0
        
        return ious