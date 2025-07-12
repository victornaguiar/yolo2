#!/usr/bin/env python
"""Script to run tracking on a dataset."""

import argparse
import os
import sys
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import MOTDataLoader
from src.tracking import BotSortTracker, YOLOTracker
from src.utils.file_utils import ensure_dir
import numpy as np


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run tracking on MOT dataset')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--tracker', type=str, default='botsort',
                       choices=['botsort', 'yolo'],
                       help='Tracking algorithm to use')
    parser.add_argument('--sequences', type=str, nargs='+', default=None,
                       help='Specific sequences to process (default: all)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='Confidence threshold for detections')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    ensure_dir(str(output_dir))
    
    # Initialize data loader
    print(f"Loading dataset from: {args.dataset}")
    try:
        data_loader = MOTDataLoader(args.dataset)
    except ValueError as e:
        print(f"Error loading dataset: {e}")
        return 1
    
    # Get sequences to process
    if args.sequences:
        sequences = args.sequences
    else:
        sequences = data_loader.get_sequence_list()
    
    if not sequences:
        print("No sequences found to process.")
        return 1
    
    print(f"Found {len(sequences)} sequences to process: {sequences}")
    
    # Initialize tracker
    print(f"Initializing {args.tracker} tracker...")
    if args.tracker == 'botsort':
        try:
            tracker = BotSortTracker(
                device=args.device if args.device != 'auto' else 'cpu',
                with_reid=False
            )
        except ImportError as e:
            print(f"Error initializing BotSort tracker: {e}")
            print("Make sure boxmot is installed: pip install boxmot")
            return 1
    elif args.tracker == 'yolo':
        tracker = YOLOTracker(
            confidence=args.confidence,
            device=args.device
        )
    
    # Process each sequence
    total_start_time = time.time()
    
    for i, seq_name in enumerate(sequences):
        print(f"\n[{i+1}/{len(sequences)}] Processing sequence: {seq_name}")
        start_time = time.time()
        
        try:
            if args.tracker == 'botsort':
                # For BotSort, we need detections file
                detection_file = data_loader.detections_dir / f"{seq_name}.txt"
                if not detection_file.exists():
                    print(f"Warning: Detection file not found for {seq_name}. Skipping.")
                    continue
                
                detections_by_frame = data_loader.load_detections(str(detection_file))
                
                # Reset tracker for new sequence
                tracker.reset()
                
                # Load and process frames
                frames = data_loader.load_sequence_frames(seq_name)
                if not frames:
                    print(f"Warning: No frames found for {seq_name}. Skipping.")
                    continue
                
                all_results = []
                
                for frame_num, frame_img in frames:
                    # Get detections for current frame
                    current_detections = detections_by_frame.get(frame_num, [])
                    detections_np = np.array(current_detections) if current_detections else np.array([])
                    
                    # Update tracker
                    if len(detections_np) > 0:
                        tracks = tracker.update(detections_np, frame_img)
                        
                        # Save tracking results
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
                output_file = output_dir / f"{seq_name}.txt"
                with open(output_file, 'w') as f:
                    f.writelines(all_results)
            
            elif args.tracker == 'yolo':
                # For YOLO, we need to reconstruct video from frames
                frames = data_loader.load_sequence_frames(seq_name)
                if not frames:
                    print(f"Warning: No frames found for {seq_name}. Skipping.")
                    continue
                
                # Reset tracker for new sequence
                tracker.reset()
                
                all_results = []
                
                for frame_num, frame_img in frames:
                    # Run YOLO tracking
                    tracks = tracker.update(None, frame_img)
                    
                    # Save tracking results
                    for track in tracks:
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
                output_file = output_dir / f"{seq_name}.txt"
                with open(output_file, 'w') as f:
                    f.writelines(all_results)
            
            end_time = time.time()
            print(f"Completed in {end_time - start_time:.2f} seconds")
            print(f"Results saved to: {output_file}")
            
        except Exception as e:
            print(f"Error processing sequence {seq_name}: {e}")
            continue
    
    total_end_time = time.time()
    print(f"\nAll sequences processed in {total_end_time - total_start_time:.2f} seconds")
    print(f"Results saved in: {output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())