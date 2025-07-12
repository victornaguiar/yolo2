#!/usr/bin/env python
"""Script to download required models."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.file_utils import download_file, ensure_dir
from config.paths import MODELS_DIR, YOLO_MODEL_PATH


def download_yolo_models():
    """Download YOLO models."""
    ensure_dir(str(MODELS_DIR))
    
    models = {
        'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
        'yolov8s.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
        'yolov8m.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
        'yolov8l.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt',
        'yolov8x.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt',
    }
    
    print("Downloading YOLO models...")
    
    for model_name, url in models.items():
        model_path = MODELS_DIR / model_name
        
        if model_path.exists():
            print(f"Model {model_name} already exists. Skipping.")
            continue
        
        print(f"Downloading {model_name}...")
        success = download_file(url, str(model_path))
        
        if success:
            print(f"✓ Downloaded {model_name}")
        else:
            print(f"✗ Failed to download {model_name}")


def download_sample_data():
    """Download sample videos and datasets."""
    from config.paths import DATA_DIR
    
    sample_videos_dir = DATA_DIR / "sample_videos"
    ensure_dir(str(sample_videos_dir))
    
    # Download sample video
    sample_video_url = "https://github.com/mikel-brostrom/yolov8_tracking/raw/main/data/videos/people.mp4"
    sample_video_path = sample_videos_dir / "people.mp4"
    
    if not sample_video_path.exists():
        print("Downloading sample video...")
        success = download_file(sample_video_url, str(sample_video_path))
        if success:
            print("✓ Downloaded sample video")
        else:
            print("✗ Failed to download sample video")
    else:
        print("Sample video already exists. Skipping.")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download models and sample data')
    parser.add_argument('--models', action='store_true', help='Download YOLO models')
    parser.add_argument('--data', action='store_true', help='Download sample data')
    parser.add_argument('--all', action='store_true', help='Download everything')
    
    args = parser.parse_args()
    
    if args.all or not any([args.models, args.data]):
        # Download everything if no specific option is given
        download_yolo_models()
        download_sample_data()
    else:
        if args.models:
            download_yolo_models()
        if args.data:
            download_sample_data()
    
    print("\nDownload complete!")


if __name__ == '__main__':
    main()