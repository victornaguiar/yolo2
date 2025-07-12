"""Tracking configuration for the soccer tracking pipeline."""

# YOLO Configuration
YOLO_CONFIG = {
    "model_name": "yolov8n.pt",
    "confidence": 0.3,
    "iou_threshold": 0.7,
    "device": "auto",  # auto, cpu, cuda:0, etc.
    "classes": [0],  # person class only
}

# BotSort Configuration
BOTSORT_CONFIG = {
    "track_thresh": 0.25,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "mot20": False,
    "with_reid": False,
    "fast_reid_config": None,
    "fast_reid_weights": None,
}

# Tracking Parameters
TRACKING_CONFIG = {
    "max_age": 70,
    "min_hits": 3,
    "iou_threshold": 0.3,
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    "metrics": ["HOTA", "CLEAR", "Identity"],
    "threshold": 0.5,
}

# Video Processing Configuration
VIDEO_CONFIG = {
    "fps": 30,
    "codec": "mp4v",
    "quality": 90,
}