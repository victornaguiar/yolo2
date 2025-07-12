"""Path configuration for the soccer tracking pipeline."""

import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Google Drive paths (for Colab)
GDRIVE_PATH = "/content/drive/MyDrive"
GDRIVE_DATASET_PATH = os.path.join(GDRIVE_PATH, "SOCCER_DATA/deepsort_dataset_train")

# Local paths
LOCAL_DATASET_PATH = "/content/soccer_dataset"
LOCAL_RESULTS_DIR = "/content/tracking_results"

# Dataset subdirectories
SEQUENCES_DIR = "sequences"
DETECTIONS_DIR = "detections"
TRACKING_RESULTS_DIR = "tracking_results"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
YOLO_MODEL_PATH = MODELS_DIR / "yolov8n.pt"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Sample data
SAMPLE_VIDEO_URL = "https://github.com/mikel-brostrom/yolov8_tracking/raw/main/data/videos/people.mp4"