#!/bin/bash
# Setup script for Google Colab environment

echo "Setting up soccer tracking pipeline in Google Colab..."

# Install system dependencies
apt-get update -qq
apt-get install -y ffmpeg

# Install Python dependencies
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
pip install ultralytics boxmot
pip install git+https://github.com/JonathonLuiten/TrackEval.git
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/{sample_videos,detections,sequences,tracking_results}
mkdir -p models
mkdir -p output

# Download sample data
echo "Downloading sample video..."
wget -q https://github.com/mikel-brostrom/yolov8_tracking/raw/main/data/videos/people.mp4 -O data/sample_videos/people.mp4

# Set up SSH for remote development (optional)
echo "Installing colab-ssh for VS Code remote development..."
pip install colab-ssh -q

echo "Setup complete! You can now:"
echo "1. Run the simple tracking demo: python -m notebooks.02_simple_tracking_demo"
echo "2. Process soccer datasets: python scripts/run_tracking.py"
echo "3. Evaluate results: python scripts/evaluate_results.py"
echo ""
echo "For VS Code remote development, run:"
echo "from colab_ssh import launch_ssh_cloudflared"
echo "launch_ssh_cloudflared(password='your_password_here')"