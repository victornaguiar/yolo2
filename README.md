# Soccer Player Tracking Pipeline

A simple and effective computer vision pipeline for tracking soccer players using YOLO and Norfair. This project processes soccer videos and generates tracking visualizations with bounding boxes around players.

## 🚀 Quick Start

This project is designed to run in **Google Colab** for the best experience. Simply run the cells in `Master_Project.ipynb` in order, and you'll get a video with player tracking at the end!

### How to Use:

1. **Open in Google Colab**: Upload `Master_Project.ipynb` to Google Colab
2. **Run cells in order**: Execute each cell from top to bottom
3. **Get your tracking video**: The final cell produces a video with tracked players

That's it! The notebook handles all the setup, data downloading, and processing automatically.

## 📋 What the Notebook Does

When you run all cells in order, the notebook will:

1. **Install Dependencies** - Automatically installs PyTorch, YOLO, Norfair, and other required libraries
2. **Download Dataset** - Downloads the SoccerNet MOT dataset (SNMOT-062 sequence)
3. **Process Video** - Runs player tracking on soccer footage using Norfair tracker
4. **Create Output Video** - Generates a final MP4 video with:
   - **Red boxes**: Tracker predictions
   - **Green boxes**: Ground truth (actual player positions)
   - **Track IDs**: Numbers identifying each player

## 🎯 Expected Output

After running all cells, you'll get:
- **Processing confirmation**: Status messages showing frames processed (e.g., "750 frames in 2.61s")
- **Final video**: Located at `/content/tracking_video_SNMOT-062_FINAL.mp4`
- **Performance stats**: FPS and processing speed information

The output video shows soccer players with bounding boxes and track IDs, demonstrating how well the tracking algorithm follows players across frames.

## 📁 Project Structure

```
yolo2/
├── Master_Project.ipynb    # Main notebook - run this!
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── config/               # Configuration files
├── src/                  # Source code modules
├── notebooks/            # Additional demo notebooks
└── scripts/              # Utility scripts
```

## 🔧 Requirements

The notebook automatically installs all required dependencies, including:

- **PyTorch** (with CUDA support for GPU acceleration)
- **Ultralytics** (YOLOv8 for object detection)
- **Norfair** (for object tracking)
- **OpenCV** (for video processing)
- **Other utilities** (NumPy, Pandas, Matplotlib, etc.)

## 💻 System Requirements

**Recommended Environment:**
- **Google Colab** (Free or Pro) - Provides GPU acceleration
- **RAM**: 12GB+ (Colab Pro recommended for larger datasets)
- **Storage**: ~25GB free space (for dataset download)
- **GPU**: Tesla T4 or better (automatically available in Colab)

**Local Installation:**
If you prefer to run locally, install dependencies:
```bash
pip install -r requirements.txt
```

## 🎬 Demo Notebooks

Additional notebooks for specific use cases:

- `notebooks/01_setup_environment.ipynb` - Environment setup and testing
- `notebooks/02_simple_tracking_demo.ipynb` - Basic tracking demonstration

## 📊 Key Features

- **Automatic Setup**: No manual configuration required
- **GPU Acceleration**: Automatically uses available GPU in Colab
- **Multiple Object Tracking**: Tracks multiple players simultaneously
- **Visual Output**: Generates annotated videos with bounding boxes
- **Performance Monitoring**: Shows processing speed and statistics
- **MOT Format Support**: Works with standard multi-object tracking datasets

## 🔍 Understanding the Results

The output video contains:

- **Red Bounding Boxes**: Predictions from the tracking algorithm
- **Green Bounding Boxes**: Ground truth (actual correct positions)
- **Track IDs**: Numbers that identify and follow individual players
- **Consistency**: Good tracking maintains the same ID for each player across frames

## 🤝 Contributing

1. Fork the repository
2. Make your changes
3. Test with the Master_Project.ipynb notebook
4. Submit a pull request

## 📜 License

This project is open source. Feel free to use and modify for your research or projects.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [Norfair](https://github.com/tryolabs/norfair) for object tracking
- [SoccerNet](https://www.soccer-net.org/) for the dataset
- Google Colab for providing free GPU resources