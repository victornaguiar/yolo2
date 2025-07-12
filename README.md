# YOLO2

A comprehensive computer vision pipeline for tracking soccer players using YOLOv8 and various tracking algorithms (BotSort, DeepSort). Designed for Google Colab with remote VM access via VS Code's Remote-SSH.

## Features

- **Multi-Object Tracking**: Track multiple players across video frames
- **Multiple Tracking Algorithms**: Support for BotSort and built-in YOLO tracking
- **Evaluation Framework**: MOT (Multi-Object Tracking) metrics evaluation
- **Google Colab Support**: Optimized for running in Google Colab environment with remote VM access
- **Flexible Data Loading**: Support for MOT format datasets
- **Remote Development**: VS Code Remote-SSH integration for seamless development
- **Hardware Acceleration**: Automatic GPU/multi-GPU utilization

## Installation

### Google Colab Setup (Recommended)

```python
# 1. Clone the repository
git clone https://github.com/victornaguiar/yolo2.git
cd yolo2

# 2. Run the setup script
bash scripts/setup_colab.sh

# 3. Mount Google Drive for data access
from google.colab import drive
drive.mount('/content/drive')
```

### Local Installation

```bash
git clone https://github.com/victornaguiar/yolo2.git
cd yolo2
pip install -r requirements.txt
```

## Remote Development Setup

This project is optimized for remote development using VS Code's Remote-SSH extension:

### 1. Setup SSH Access (In Colab)

```python
# Install and configure SSH
!pip install colab-ssh -q

from colab_ssh import launch_ssh_cloudflared
launch_ssh_cloudflared(password="your_secure_password")
```

### 2. Connect VS Code

1. Install the "Remote - SSH" extension in VS Code
2. Use the connection details provided by the SSH setup
3. Open the project folder on the remote VM
4. Enjoy full IDE functionality with VM's GPU acceleration

## Quick Start

### 1. Simple Tracking Demo

```python
from src.tracking import YOLOTracker

# Initialize tracker
tracker = YOLOTracker(model_name='yolov8n.pt', device='auto')

# Run tracking on video
results = tracker.track_video('path/to/video.mp4', output_path='tracked_video.mp4')
```

### 2. Soccer Dataset Tracking

```python
from src.tracking import BotSortTracker
from src.data import MOTDataLoader

# Load soccer dataset
data_loader = MOTDataLoader(dataset_path='/content/soccer_dataset')

# Initialize BotSort tracker
tracker = BotSortTracker(device='auto')

# Process all sequences
tracker.process_dataset(data_loader, output_dir='tracking_results')
```

### 3. Evaluation

```python
from src.evaluation import MOTEvaluator

# Evaluate tracking results
evaluator = MOTEvaluator()
metrics = evaluator.evaluate(
    gt_dir='data/detections',
    results_dir='data/tracking_results'
)
```

## Project Structure

```
yolo2/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── config/                     # Configuration files
│   ├── __init__.py
│   ├── paths.py               # Path configurations
│   └── tracking_config.py     # Tracking parameters
├── src/                       # Core implementation
│   ├── __init__.py
│   ├── data/                  # Data loading utilities
│   │   ├── __init__.py
│   │   ├── data_loader.py     # MOT format data loader
│   │   └── video_generator.py # Video processing utilities
│   ├── tracking/              # Tracking algorithms
│   │   ├── __init__.py
│   │   ├── base_tracker.py    # Base tracker class
│   │   ├── yolo_tracker.py    # YOLO-based tracking
│   │   └── botsort_tracker.py # BotSort implementation
│   ├── evaluation/            # Evaluation framework
│   │   ├── __init__.py
│   │   ├── metrics.py         # Tracking metrics
│   │   └── mot_evaluator.py   # MOT evaluation
│   └── utils/                 # Utility functions
│       ├── __init__.py
│       ├── file_utils.py      # File operations
│       └── visualization.py   # Visualization tools
├── notebooks/                 # Jupyter notebooks
│   ├── 01_setup_environment.ipynb
│   ├── 02_simple_tracking_demo.ipynb
│   ├── 03_soccer_tracking_pipeline.ipynb
│   └── 04_evaluation_analysis.ipynb
├── scripts/                   # Command-line scripts
│   ├── setup_colab.sh         # Colab environment setup
│   ├── download_models.py     # Download pre-trained models
│   ├── run_tracking.py        # Run tracking on datasets
│   └── evaluate_results.py    # Evaluate tracking results
├── tests/                     # Unit tests
│   └── __init__.py
└── data/                      # Data directory
    ├── sample_videos/
    ├── detections/
    ├── sequences/
    └── tracking_results/
```

## Notebooks

### 1. [01_setup_environment.ipynb](notebooks/01_setup_environment.ipynb)
- Environment setup and dependency installation
- Google Drive mounting and data copying
- SSH setup for remote development
- Hardware verification and testing

### 2. [02_simple_tracking_demo.ipynb](notebooks/02_simple_tracking_demo.ipynb)
- Basic tracking demonstration
- YOLO tracker usage
- Performance benchmarking
- Video visualization

### 3. [03_soccer_tracking_pipeline.ipynb](notebooks/03_soccer_tracking_pipeline.ipynb)
- Full soccer tracking pipeline
- MOT dataset processing
- Multiple tracker comparison
- Results analysis

### 4. [04_evaluation_analysis.ipynb](notebooks/04_evaluation_analysis.ipynb)
- Results evaluation and visualization
- Metric calculation and interpretation
- Performance analysis and optimization

## Command-Line Usage

### Download Models and Data

```bash
# Download all models and sample data
python scripts/download_models.py --all

# Download only YOLO models
python scripts/download_models.py --models

# Download only sample data
python scripts/download_models.py --data
```

### Run Tracking

```bash
# Track with BotSort
python scripts/run_tracking.py \
    --dataset /path/to/dataset \
    --output /path/to/results \
    --tracker botsort \
    --device auto

# Track with YOLO
python scripts/run_tracking.py \
    --dataset /path/to/dataset \
    --output /path/to/results \
    --tracker yolo \
    --confidence 0.3
```

### Evaluate Results

```bash
# Evaluate tracking results
python scripts/evaluate_results.py \
    --gt_dir /path/to/ground_truth \
    --results_dir /path/to/tracking_results \
    --output evaluation_results.json
```

## Configuration

### Path Configuration

Edit `config/paths.py` to set your data paths:

```python
# Google Drive paths (for Colab)
GDRIVE_DATASET_PATH = "/content/drive/MyDrive/SOCCER_DATA/dataset"

# Local paths
LOCAL_DATASET_PATH = "/content/soccer_dataset"
LOCAL_RESULTS_DIR = "/content/tracking_results"
```

### Tracking Configuration

Edit `config/tracking_config.py` for tracking parameters:

```python
# YOLO Configuration
YOLO_CONFIG = {
    "model_name": "yolov8n.pt",
    "confidence": 0.3,
    "device": "auto",
    "classes": [0],  # person class only
}

# BotSort Configuration
BOTSORT_CONFIG = {
    "track_thresh": 0.25,
    "track_buffer": 30,
    "with_reid": False,
}
```

## Remote VM Workflow

This pipeline is designed for the following workflow:

1. **Development**: Use VS Code with Remote-SSH on your local machine
2. **Data Storage**: Keep original datasets on Google Drive
3. **Processing**: Copy data to VM's SSD for fast access during processing
4. **Computation**: Leverage VM's GPU/multi-GPU for acceleration
5. **Results**: Save results back to Google Drive for persistence

### Advantages:
- **Fast Processing**: VM's SSD and GPU acceleration
- **Full IDE**: Complete VS Code functionality remotely
- **Data Persistence**: Google Drive backup
- **Scalability**: Easy to upgrade VM resources
- **Cost Effective**: Pay only for computation time

## Dependencies

- **Python 3.8+**
- **PyTorch** (with CUDA support)
- **Ultralytics (YOLOv8)**
- **BoxMOT** (for BotSort)
- **OpenCV**
- **NumPy, SciPy**
- **TrackEval** (for evaluation)
- **Matplotlib, Pandas** (for visualization)

## Hardware Requirements

### Minimum:
- **RAM**: 8GB
- **Storage**: 10GB free space
- **GPU**: Any CUDA-compatible GPU (optional but recommended)

### Recommended (Google Colab Pro):
- **RAM**: 16GB+
- **GPU**: Tesla T4, V100, or A100
- **Storage**: High-performance SSD

## Performance Optimization

### For Real-time Processing:
- Use `yolov8n.pt` for fastest inference
- Set appropriate confidence thresholds
- Use GPU acceleration (`device='auto'`)

### For Best Accuracy:
- Use `yolov8x.pt` for highest accuracy
- Enable ReID features in BotSort
- Lower confidence thresholds for more detections

### Memory Optimization:
- Process videos in batches
- Use smaller input resolutions
- Clear tracking history periodically

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{yolo2,
  author = {Victor Naguiar},
  title = {YOLO2},
  year = {2024},
  url = {https://github.com/victornaguiar/yolo2}
}
```

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [BoxMOT](https://github.com/mikel-brostrom/boxmot)
- [TrackEval](https://github.com/JonathonLuiten/TrackEval)
- [Google Colab](https://colab.research.google.com/)

## Support

For questions and support:
- 📧 Email: victornaguiar@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/victornaguiar/yolo2/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/victornaguiar/yolo2/discussions)