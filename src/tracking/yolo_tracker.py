"""YOLO tracker implementation with built-in tracking."""

import numpy as np
import torch
import cv2
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Optional
import os

from .base_tracker import BaseTracker
from ..utils.file_utils import ensure_dir


class YOLOTracker(BaseTracker):
    """YOLO object detection and tracking."""
    
    def __init__(self, 
                 model_name: str = 'yolov8n.pt',
                 confidence: float = 0.3,
                 iou_threshold: float = 0.7,
                 device: str = 'auto',
                 classes: Optional[List[int]] = None):
        """
        Initialize YOLO tracker.
        
        Args:
            model_name: YOLO model name or path
            confidence: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run on ('auto', 'cpu', 'cuda', etc.)
            classes: List of class IDs to track (None = all classes)
        """
        super().__init__()
        
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.classes = classes or [0]  # Default to person class
        
        # Determine device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load YOLO model
        self.model = YOLO(model_name)
        self.model.to(self.device)
        
        print(f"YOLO tracker initialized with {model_name} on {self.device}")
    
    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        Update tracker with new detections using YOLO's built-in tracker.
        
        Args:
            detections: Pre-computed detections (ignored, YOLO does its own detection)
            frame: Current frame
            
        Returns:
            Array of tracks [x1, y1, x2, y2, track_id, conf, cls]
        """
        # Run YOLO tracking on the frame
        results = self.model.track(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=self.classes,
            persist=True,
            verbose=False
        )
        
        tracks = []
        
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            
            # Extract tracking information
            if hasattr(boxes, 'id') and boxes.id is not None:
                # Tracking IDs are available
                for i in range(len(boxes)):
                    # Get box coordinates
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    
                    # Get confidence
                    conf = boxes.conf[i].cpu().numpy()
                    
                    # Get class
                    cls = boxes.cls[i].cpu().numpy()
                    
                    # Get track ID
                    track_id = boxes.id[i].cpu().numpy()
                    
                    # Create track entry
                    track = np.array([xyxy[0], xyxy[1], xyxy[2], xyxy[3], track_id, conf, cls])
                    tracks.append(track)
        
        return np.array(tracks) if tracks else np.array([])
    
    def detect_only(self, frame: np.ndarray) -> np.ndarray:
        """
        Run detection only (no tracking).
        
        Args:
            frame: Input frame
            
        Returns:
            Array of detections [x1, y1, x2, y2, conf, cls]
        """
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=self.classes,
            verbose=False
        )
        
        detections = []
        
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                # Get box coordinates
                xyxy = boxes.xyxy[i].cpu().numpy()
                
                # Get confidence
                conf = boxes.conf[i].cpu().numpy()
                
                # Get class
                cls = boxes.cls[i].cpu().numpy()
                
                # Create detection entry
                detection = np.array([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, cls])
                detections.append(detection)
        
        return np.array(detections) if detections else np.array([])
    
    def track_video(self, 
                    video_path: str, 
                    output_path: Optional[str] = None,
                    save_frames: bool = False,
                    frame_output_dir: Optional[str] = None) -> List[Dict]:
        """
        Track objects in a video.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            save_frames: Whether to save individual frames
            frame_output_dir: Directory to save frames (if save_frames=True)
            
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
            ensure_dir(os.path.dirname(output_path))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Initialize frame output directory
        if save_frames and frame_output_dir:
            ensure_dir(frame_output_dir)
        
        results = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update tracker
            tracks = self.update(None, frame)
            
            # Store results
            frame_results = {
                'frame_idx': frame_idx,
                'tracks': tracks.tolist() if len(tracks) > 0 else []
            }
            results.append(frame_results)
            
            # Draw tracks on frame if output is requested
            if output_path or save_frames:
                annotated_frame = self.draw_tracks(frame.copy(), tracks)
                
                if out:
                    out.write(annotated_frame)
                
                if save_frames and frame_output_dir:
                    frame_filename = f"frame_{frame_idx:06d}.jpg"
                    frame_path = os.path.join(frame_output_dir, frame_filename)
                    cv2.imwrite(frame_path, annotated_frame)
            
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
            x1, y1, x2, y2, track_id, conf, cls = track
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
    
    def reset(self):
        """Reset the tracker state."""
        self.frame_count = 0
        self.track_history = {}
        # YOLO tracker reset is handled internally when persist=True is used