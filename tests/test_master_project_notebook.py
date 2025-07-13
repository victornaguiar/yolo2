"""Test the Master_Project.ipynb notebook creation."""

import unittest
import json
import os
from pathlib import Path


class TestMasterProjectNotebook(unittest.TestCase):
    """Test cases for Master_Project.ipynb notebook."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.repo_root = Path(__file__).parent.parent
        self.notebooks_dir = self.repo_root / "notebooks"
        self.master_notebook_path = self.repo_root / "Master_Project.ipynb"
        
        # The four source notebooks that should be merged
        self.source_notebooks = [
            "01_setup_environment.ipynb",
            "02_simple_tracking_demo.ipynb",
            "03_soccer_tracking_pipeline.ipynb",
            "04_evaluation_analysis.ipynb"
        ]
    
    def test_master_notebook_exists(self):
        """Test that Master_Project.ipynb exists in the root directory."""
        self.assertTrue(self.master_notebook_path.exists(), 
                       f"Master_Project.ipynb should exist at {self.master_notebook_path}")
    
    def test_master_notebook_is_valid_json(self):
        """Test that Master_Project.ipynb is valid JSON."""
        with open(self.master_notebook_path, 'r', encoding='utf-8') as f:
            try:
                notebook_data = json.load(f)
                self.assertIsInstance(notebook_data, dict)
            except json.JSONDecodeError as e:
                self.fail(f"Master_Project.ipynb is not valid JSON: {e}")
    
    def test_master_notebook_has_correct_structure(self):
        """Test that Master_Project.ipynb has the correct Jupyter notebook structure."""
        with open(self.master_notebook_path, 'r', encoding='utf-8') as f:
            notebook_data = json.load(f)
        
        # Check required keys
        required_keys = ['cells', 'metadata', 'nbformat', 'nbformat_minor']
        for key in required_keys:
            self.assertIn(key, notebook_data, f"Missing required key: {key}")
        
        # Check that cells is a list
        self.assertIsInstance(notebook_data['cells'], list)
        
        # Check notebook format
        self.assertEqual(notebook_data['nbformat'], 4)
        self.assertIsInstance(notebook_data['nbformat_minor'], int)
    
    def test_master_notebook_contains_all_source_content(self):
        """Test that Master_Project.ipynb contains content from all four source notebooks."""
        # Read the master notebook
        with open(self.master_notebook_path, 'r', encoding='utf-8') as f:
            master_notebook = json.load(f)
        
        # Read all source notebooks and count their cells
        total_expected_cells = 0
        expected_titles = []
        
        for notebook_name in self.source_notebooks:
            notebook_path = self.notebooks_dir / notebook_name
            if notebook_path.exists():
                with open(notebook_path, 'r', encoding='utf-8') as f:
                    source_notebook = json.load(f)
                    total_expected_cells += len(source_notebook.get('cells', []))
                    
                    # Look for title cells
                    for cell in source_notebook.get('cells', []):
                        if cell.get('cell_type') == 'markdown':
                            source = ''.join(cell.get('source', []))
                            if source.startswith('# '):
                                expected_titles.append(source.split('\n')[0])
                                break
        
        # Check that the master notebook has the expected number of cells
        self.assertEqual(len(master_notebook['cells']), total_expected_cells,
                        f"Expected {total_expected_cells} cells, got {len(master_notebook['cells'])}")
        
        # Check that titles from all source notebooks are present
        master_titles = []
        for cell in master_notebook['cells']:
            if cell.get('cell_type') == 'markdown':
                source = ''.join(cell.get('source', []))
                if source.startswith('# '):
                    master_titles.append(source.split('\n')[0])
        
        for expected_title in expected_titles:
            self.assertIn(expected_title, master_titles,
                         f"Expected title '{expected_title}' not found in master notebook")
    
    def test_source_notebooks_still_exist(self):
        """Test that the original source notebooks still exist and were not deleted."""
        for notebook_name in self.source_notebooks:
            notebook_path = self.notebooks_dir / notebook_name
            self.assertTrue(notebook_path.exists(),
                           f"Source notebook {notebook_name} should still exist at {notebook_path}")
    
    def test_master_notebook_cells_are_valid(self):
        """Test that all cells in Master_Project.ipynb are valid Jupyter cells."""
        with open(self.master_notebook_path, 'r', encoding='utf-8') as f:
            notebook_data = json.load(f)
        
        for i, cell in enumerate(notebook_data['cells']):
            # Check that each cell has required fields
            self.assertIn('cell_type', cell, f"Cell {i} missing cell_type")
            self.assertIn('source', cell, f"Cell {i} missing source")
            self.assertIn('metadata', cell, f"Cell {i} missing metadata")
            
            # Check valid cell types
            valid_cell_types = ['markdown', 'code', 'raw']
            self.assertIn(cell['cell_type'], valid_cell_types,
                         f"Cell {i} has invalid cell_type: {cell['cell_type']}")
            
            # Check that source is a list
            self.assertIsInstance(cell['source'], list,
                                f"Cell {i} source should be a list")


if __name__ == '__main__':
    unittest.main()