"""Test the data loading functionality."""

import unittest
import tempfile
import os
from pathlib import Path
import numpy as np
import cv2

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data import MOTDataLoader


class TestMOTDataLoader(unittest.TestCase):
    """Test cases for MOTDataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = Path(self.temp_dir)
        
        # Create test directory structure
        (self.dataset_path / "sequences" / "test_seq").mkdir(parents=True)
        (self.dataset_path / "detections").mkdir(parents=True)
        
        # Create test detection file
        detection_data = [
            "1,1,100,100,50,50,0.9,1,1,1",
            "1,2,200,200,60,60,0.8,1,1,1",
            "2,1,110,110,50,50,0.9,1,1,1",
            "2,3,300,300,40,40,0.7,1,1,1"
        ]
        
        detection_file = self.dataset_path / "detections" / "test_seq.txt"
        with open(detection_file, 'w') as f:
            f.write('\n'.join(detection_data))
        
        # Create test frame (simple image)
        test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cv2.imwrite(str(self.dataset_path / "sequences" / "test_seq" / "1.jpg"), test_frame)
        cv2.imwrite(str(self.dataset_path / "sequences" / "test_seq" / "2.jpg"), test_frame)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test MOTDataLoader initialization."""
        loader = MOTDataLoader(str(self.dataset_path))
        self.assertEqual(loader.dataset_path, self.dataset_path)
        self.assertTrue(loader.sequences_dir.exists())
        self.assertTrue(loader.detections_dir.exists())
    
    def test_invalid_path(self):
        """Test initialization with invalid path."""
        with self.assertRaises(ValueError):
            MOTDataLoader("/nonexistent/path")
    
    def test_load_detections(self):
        """Test loading detections from file."""
        loader = MOTDataLoader(str(self.dataset_path))
        detection_file = self.dataset_path / "detections" / "test_seq.txt"
        
        detections = loader.load_detections(str(detection_file))
        
        # Check frame 1 detections
        self.assertIn(1, detections)
        self.assertEqual(len(detections[1]), 2)  # 2 detections in frame 1
        
        # Check frame 2 detections
        self.assertIn(2, detections)
        self.assertEqual(len(detections[2]), 2)  # 2 detections in frame 2
        
        # Check detection format [x1, y1, x2, y2, confidence, class_id]
        det = detections[1][0]
        self.assertEqual(len(det), 6)
        self.assertEqual(det[0], 100)  # x1
        self.assertEqual(det[1], 100)  # y1
        self.assertEqual(det[2], 150)  # x2 = x1 + width
        self.assertEqual(det[3], 150)  # y2 = y1 + height
    
    def test_get_sequence_list(self):
        """Test getting list of sequences."""
        loader = MOTDataLoader(str(self.dataset_path))
        sequences = loader.get_sequence_list()
        
        self.assertIn("test_seq", sequences)
        self.assertEqual(len(sequences), 1)
    
    def test_load_sequence_frames(self):
        """Test loading frames from a sequence."""
        loader = MOTDataLoader(str(self.dataset_path))
        frames = loader.load_sequence_frames("test_seq")
        
        self.assertEqual(len(frames), 2)  # 2 frames
        
        # Check frame data
        frame_num, frame_img = frames[0]
        self.assertEqual(frame_num, 1)
        self.assertIsInstance(frame_img, np.ndarray)
        self.assertEqual(frame_img.shape, (480, 640, 3))
    
    def test_get_sequence_info(self):
        """Test getting sequence information."""
        loader = MOTDataLoader(str(self.dataset_path))
        info = loader.get_sequence_info("test_seq")
        
        self.assertEqual(info['name'], "test_seq")
        self.assertIn('length', info)
        self.assertIn('width', info)
        self.assertIn('height', info)
        self.assertIn('fps', info)


if __name__ == '__main__':
    unittest.main()