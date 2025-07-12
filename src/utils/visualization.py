"""Visualization utilities for the soccer tracking pipeline."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional, Dict
import os


def draw_tracks(frame: np.ndarray, 
               tracks: np.ndarray,
               track_history: Optional[Dict] = None,
               show_trajectory: bool = False,
               trajectory_length: int = 30) -> np.ndarray:
    """
    Draw tracking results on frame.
    
    Args:
        frame: Input frame
        tracks: Array of tracks [x1, y1, x2, y2, track_id, conf, cls, ...]
        track_history: History of track positions
        show_trajectory: Whether to show track trajectories
        trajectory_length: Number of past positions to show
        
    Returns:
        Annotated frame
    """
    annotated_frame = frame.copy()
    
    # Draw trajectories if requested
    if show_trajectory and track_history:
        for track in tracks:
            if len(track) >= 5:
                track_id = int(track[4])
                if track_id in track_history:
                    # Get recent positions
                    positions = track_history[track_id][-trajectory_length:]
                    if len(positions) > 1:
                        # Draw trajectory
                        color = ((track_id * 50) % 255, (track_id * 100) % 255, (track_id * 150) % 255)
                        for i in range(1, len(positions)):
                            pt1 = (int(positions[i-1][0]), int(positions[i-1][1]))
                            pt2 = (int(positions[i][0]), int(positions[i][1]))
                            cv2.line(annotated_frame, pt1, pt2, color, 2)
    
    # Draw current tracks
    for track in tracks:
        if len(track) >= 7:
            x1, y1, x2, y2, track_id, conf, cls = track[:7]
        else:
            continue
        
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        track_id = int(track_id)
        
        # Choose color based on track ID
        color = ((track_id * 50) % 255, (track_id * 100) % 255, (track_id * 150) % 255)
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw track ID and confidence
        label = f"ID: {track_id} ({conf:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(annotated_frame, (center_x, center_y), 3, color, -1)
    
    return annotated_frame


def create_video_from_frames(frames: List[np.ndarray], 
                           output_path: str,
                           fps: int = 30,
                           codec: str = 'mp4v') -> bool:
    """
    Create video from list of frames.
    
    Args:
        frames: List of frame images
        output_path: Output video path
        fps: Frames per second
        codec: Video codec
        
    Returns:
        True if successful, False otherwise
    """
    if not frames:
        print("No frames provided")
        return False
    
    # Get frame dimensions
    h, w = frames[0].shape[:2]
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        return False
    
    # Write frames
    for frame in frames:
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        out.write(frame)
    
    out.release()
    print(f"Video saved to: {output_path}")
    return True


def plot_tracking_statistics(track_history: Dict, 
                           output_path: Optional[str] = None) -> plt.Figure:
    """
    Plot tracking statistics.
    
    Args:
        track_history: Dictionary of track histories
        output_path: Path to save plot (optional)
        
    Returns:
        matplotlib Figure object
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Track lengths histogram
    track_lengths = [len(history) for history in track_history.values()]
    ax1.hist(track_lengths, bins=20, alpha=0.7)
    ax1.set_title('Track Length Distribution')
    ax1.set_xlabel('Track Length (frames)')
    ax1.set_ylabel('Number of Tracks')
    
    # Track count over time
    max_frame = max(max(entry['frame'] for entry in history) 
                   for history in track_history.values())
    frame_counts = [0] * (max_frame + 1)
    
    for history in track_history.values():
        for entry in history:
            frame_counts[entry['frame']] += 1
    
    ax2.plot(frame_counts)
    ax2.set_title('Active Tracks Over Time')
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('Number of Active Tracks')
    
    # Track lifetime visualization
    y_pos = 0
    for track_id, history in list(track_history.items())[:20]:  # Show first 20 tracks
        frames = [entry['frame'] for entry in history]
        ax3.plot([min(frames), max(frames)], [y_pos, y_pos], linewidth=2)
        y_pos += 1
    
    ax3.set_title('Track Lifetimes (First 20 tracks)')
    ax3.set_xlabel('Frame Number')
    ax3.set_ylabel('Track ID (relative)')
    
    # Statistics summary
    stats_text = f"""
    Total Tracks: {len(track_history)}
    Total Frames: {max_frame}
    Avg Track Length: {np.mean(track_lengths):.1f}
    Max Track Length: {max(track_lengths)}
    Min Track Length: {min(track_lengths)}
    """
    
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, 
             fontsize=12, verticalalignment='center')
    ax4.set_title('Statistics Summary')
    ax4.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    return fig


def visualize_detections(frame: np.ndarray, 
                        detections: np.ndarray,
                        confidence_threshold: float = 0.5) -> np.ndarray:
    """
    Visualize detections on frame.
    
    Args:
        frame: Input frame
        detections: Array of detections [x1, y1, x2, y2, conf, cls]
        confidence_threshold: Minimum confidence to display
        
    Returns:
        Annotated frame
    """
    annotated_frame = frame.copy()
    
    for detection in detections:
        if len(detection) >= 5:
            x1, y1, x2, y2, conf = detection[:5]
            
            if conf < confidence_threshold:
                continue
            
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence
            label = f"{conf:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return annotated_frame


def create_comparison_video(frames1: List[np.ndarray],
                           frames2: List[np.ndarray],
                           output_path: str,
                           labels: Tuple[str, str] = ("Method 1", "Method 2"),
                           fps: int = 30) -> bool:
    """
    Create side-by-side comparison video.
    
    Args:
        frames1: First set of frames
        frames2: Second set of frames
        output_path: Output video path
        labels: Labels for each method
        fps: Frames per second
        
    Returns:
        True if successful, False otherwise
    """
    if len(frames1) != len(frames2):
        print("Error: Frame lists must have the same length")
        return False
    
    if not frames1:
        print("Error: No frames provided")
        return False
    
    # Get frame dimensions
    h1, w1 = frames1[0].shape[:2]
    h2, w2 = frames2[0].shape[:2]
    
    # Use the larger height and combined width
    combined_h = max(h1, h2)
    combined_w = w1 + w2
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (combined_w, combined_h))
    
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        return False
    
    # Process frames
    for frame1, frame2 in zip(frames1, frames2):
        # Resize frames to common height
        frame1_resized = cv2.resize(frame1, (w1, combined_h))
        frame2_resized = cv2.resize(frame2, (w2, combined_h))
        
        # Add labels
        cv2.putText(frame1_resized, labels[0], (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame2_resized, labels[1], (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Combine frames horizontally
        combined_frame = np.hstack([frame1_resized, frame2_resized])
        
        # Add separator line
        cv2.line(combined_frame, (w1, 0), (w1, combined_h), (255, 255, 255), 2)
        
        out.write(combined_frame)
    
    out.release()
    print(f"Comparison video saved to: {output_path}")
    return True