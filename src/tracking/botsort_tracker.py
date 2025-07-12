"""BotSort tracker implementation."""

import numpy as np
import torch
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict

try:
    from boxmot import BotSORT
except ImportError:
    print("Warning: boxmot not available. Install with: pip install boxmot")
    BotSORT = None

from .base_tracker import BaseTracker


class BotSortTracker(BaseTracker):
    """BotSort tracking algorithm wrapper."""
    
    def __init__(self, 
                 reid_weights: Optional[str] = None,
                 device: str = 'cpu',
                 with_reid: bool = False,
                 track_thresh: float = 0.25,
                 track_buffer: int = 30,
                 match_thresh: float = 0.8,
                 mot20: bool = False):
        """
        Initialize BotSort tracker.
        
        Args:
            reid_weights: Path to ReID model weights
            device: Device to run on ('cpu' or 'cuda')
            with_reid: Whether to use ReID features
            track_thresh: Detection confidence threshold
            track_buffer: Number of frames to keep lost tracks
            match_thresh: Matching threshold
            mot20: Whether to use MOT20 evaluation protocol
        """
        super().__init__()
        
        if BotSORT is None:
            raise ImportError("boxmot is required for BotSort tracker. Install with: pip install boxmot")
        
        # Use dummy weights if none provided and ReID is disabled
        if reid_weights is None and not with_reid:
            reid_weights = "dummy_weights.pt"
        
        self.tracker = BotSORT(
            model_weights=Path(reid_weights) if reid_weights else None,
            device=torch.device(device),
            half=False,
            with_reid=with_reid,
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            mot20=mot20
        )
        
        self.device = device
        self.with_reid = with_reid
        
        print(f"BotSort tracker initialized on {device}")
    
    def update(self, 
               detections: np.ndarray, 
               frame: np.ndarray) -> np.ndarray:
        """
        Update tracker with new detections.
        
        Args:
            detections: Array of detections [x1, y1, x2, y2, conf, cls]
            frame: Current frame
            
        Returns:
            Array of tracks [x1, y1, x2, y2, track_id, conf, cls, ...]
        """
        if len(detections) == 0:
            return np.array([])
        
        # Ensure detections are in the correct format
        if detections.shape[1] < 6:
            # If no class information, assume all are class 0 (person)
            detections = np.column_stack([detections, np.zeros((detections.shape[0], 1))])
        
        # BotSort expects detections in format [x1, y1, x2, y2, conf, cls]
        tracks = self.tracker.update(detections, frame)
        
        return tracks if tracks is not None else np.array([])
    
    def reset(self):
        """Reset the tracker state."""
        super().reset()
        
        # Reinitialize the tracker
        device = self.tracker.device
        with_reid = self.with_reid
        
        # Get current tracker parameters
        track_thresh = getattr(self.tracker, 'track_thresh', 0.25)
        track_buffer = getattr(self.tracker, 'track_buffer', 30)
        match_thresh = getattr(self.tracker, 'match_thresh', 0.8)
        mot20 = getattr(self.tracker, 'mot20', False)
        
        self.tracker = BotSORT(
            model_weights=None,  # Use default weights
            device=device,
            half=False,
            with_reid=with_reid,
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            mot20=mot20
        )
    
    def track_video(self, 
                    video_path: str, 
                    output_path: Optional[str] = None,
                    detections_dict: Optional[Dict[int, List]] = None) -> List[Dict]:
        """
        Track objects in a video.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            detections_dict: Pre-computed detections by frame (optional)
            
        Returns:
            List of tracking results per frame
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer if output path is provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        results = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get detections for current frame
            if detections_dict and frame_idx in detections_dict:
                detections = np.array(detections_dict[frame_idx])
            else:
                # If no detections provided, use empty array
                detections = np.array([])
            
            # Update tracker
            tracks = self.update(detections, frame)
            
            # Store results
            frame_results = {
                'frame_idx': frame_idx,
                'tracks': tracks.tolist() if len(tracks) > 0 else []
            }
            results.append(frame_results)
            
            # Draw tracks on frame if output is requested
            if output_path and len(tracks) > 0:
                frame = self.draw_tracks(frame, tracks)
            
            if out:
                out.write(frame)
            
            frame_idx += 1
        
        # Clean up
        cap.release()
        if out:
            out.release()
        
        print(f"Processed {frame_idx} frames")
        if output_path:
            print(f"Output video saved to: {output_path}")
        
        return results
    
    def draw_tracks(self, frame: np.ndarray, tracks: np.ndarray) -> np.ndarray:
        """Draw tracking results on frame."""
        for track in tracks:
            if len(track) >= 8:
                x1, y1, x2, y2, track_id, conf, cls, _ = track[:8]
            else:
                x1, y1, x2, y2, track_id, conf, cls = track[:7]
            
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            track_id = int(track_id)
            
            # Choose color based on track ID
            color = ((track_id * 50) % 255, (track_id * 100) % 255, (track_id * 150) % 255)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID and confidence
            label = f"ID: {track_id} ({conf:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def process_dataset(self, 
                       data_loader, 
                       output_dir: str,
                       sequences: Optional[List[str]] = None):
        """
        Process a complete MOT dataset.
        
        Args:
            data_loader: MOTDataLoader instance
            output_dir: Directory to save tracking results
            sequences: List of sequences to process (None = all)
        """
        from pathlib import Path
        import os
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get sequences to process
        if sequences is None:
            sequences = data_loader.get_sequence_list()
        
        print(f"Processing {len(sequences)} sequences with BotSort")
        
        for i, seq_name in enumerate(sequences):
            print(f"[{i+1}/{len(sequences)}] Processing sequence: {seq_name}")
            
            # Reset tracker for new sequence
            self.reset()
            
            # Load detections
            detection_file = data_loader.detections_dir / f"{seq_name}.txt"
            if not detection_file.exists():
                print(f"Warning: Detection file not found for {seq_name}. Skipping.")
                continue
            
            detections_by_frame = data_loader.load_detections(str(detection_file))
            
            # Load and process frames
            frames = data_loader.load_sequence_frames(seq_name)
            all_results = []
            
            for frame_num, frame_img in frames:
                # Get detections for current frame
                current_detections = detections_by_frame.get(frame_num, [])
                detections_np = np.array(current_detections) if current_detections else np.array([])
                
                # Update tracker
                if len(detections_np) > 0:
                    tracks = self.update(detections_np, frame_img)
                    
                    # Save tracking results in MOT format
                    for track in tracks:
                        if len(track) >= 8:
                            x1, y1, x2, y2, track_id, conf, cls, _ = track[:8]
                        else:
                            x1, y1, x2, y2, track_id, conf, cls = track[:7]
                        
                        # Convert back to MOT format
                        bb_left = x1
                        bb_top = y1
                        bb_width = x2 - x1
                        bb_height = y2 - y1
                        
                        result_line = f"{frame_num},{int(track_id)},{bb_left:.2f},{bb_top:.2f}," \
                                     f"{bb_width:.2f},{bb_height:.2f},{conf:.2f},-1,-1,-1\n"
                        all_results.append(result_line)
            
            # Save results
            output_file = output_path / f"{seq_name}.txt"
            with open(output_file, 'w') as f:
                f.writelines(all_results)
            
            print(f"Results saved to: {output_file}")
        
        print(f"All sequences processed. Results saved in: {output_dir}")