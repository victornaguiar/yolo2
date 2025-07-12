"""Base tracker class for the soccer tracking pipeline."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Optional, Tuple


class BaseTracker(ABC):
    """Abstract base class for object trackers."""
    
    def __init__(self):
        """Initialize the base tracker."""
        self.frame_count = 0
        self.track_history = {}
        
    @abstractmethod
    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        Update tracker with new detections.
        
        Args:
            detections: Array of detections [x1, y1, x2, y2, conf, cls]
            frame: Current frame
            
        Returns:
            Array of tracks [x1, y1, x2, y2, track_id, conf, cls, ...]
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset the tracker state."""
        pass
    
    def process_frame(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame with detections.
        
        Args:
            detections: Array of detections
            frame: Current frame
            
        Returns:
            Array of tracks
        """
        self.frame_count += 1
        tracks = self.update(detections, frame)
        
        # Update track history
        for track in tracks:
            track_id = int(track[4])  # Assuming track_id is at index 4
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append({
                'frame': self.frame_count,
                'bbox': track[:4],
                'confidence': track[5] if len(track) > 5 else 1.0
            })
        
        return tracks
    
    def get_track_history(self, track_id: Optional[int] = None) -> Dict:
        """
        Get tracking history.
        
        Args:
            track_id: Specific track ID to get (None = all tracks)
            
        Returns:
            Dictionary of track histories
        """
        if track_id is not None:
            return {track_id: self.track_history.get(track_id, [])}
        return self.track_history
    
    def save_results(self, output_path: str, format: str = 'mot'):
        """
        Save tracking results to file.
        
        Args:
            output_path: Path to save results
            format: Output format ('mot', 'json')
        """
        if format == 'mot':
            self._save_mot_format(output_path)
        elif format == 'json':
            self._save_json_format(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_mot_format(self, output_path: str):
        """Save results in MOT format."""
        with open(output_path, 'w') as f:
            for track_id, history in self.track_history.items():
                for entry in history:
                    frame = entry['frame']
                    bbox = entry['bbox']
                    conf = entry['confidence']
                    
                    # MOT format: frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z
                    x1, y1, x2, y2 = bbox
                    width = x2 - x1
                    height = y2 - y1
                    
                    line = f"{frame},{track_id},{x1:.2f},{y1:.2f},{width:.2f},{height:.2f},{conf:.2f},-1,-1,-1\n"
                    f.write(line)
    
    def _save_json_format(self, output_path: str):
        """Save results in JSON format."""
        import json
        
        results = {
            'frame_count': self.frame_count,
            'tracks': self.track_history
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    def get_statistics(self) -> Dict:
        """
        Get tracking statistics.
        
        Returns:
            Dictionary with tracking statistics
        """
        total_tracks = len(self.track_history)
        total_detections = sum(len(history) for history in self.track_history.values())
        
        track_lengths = [len(history) for history in self.track_history.values()]
        avg_track_length = np.mean(track_lengths) if track_lengths else 0
        max_track_length = max(track_lengths) if track_lengths else 0
        min_track_length = min(track_lengths) if track_lengths else 0
        
        return {
            'total_tracks': total_tracks,
            'total_detections': total_detections,
            'frames_processed': self.frame_count,
            'avg_track_length': avg_track_length,
            'max_track_length': max_track_length,
            'min_track_length': min_track_length,
            'tracks_per_frame': total_detections / self.frame_count if self.frame_count > 0 else 0
        }