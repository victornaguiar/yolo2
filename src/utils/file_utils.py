"""File utility functions for the soccer tracking pipeline."""

import os
import shutil
import urllib.request
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm


def ensure_dir(directory: str) -> str:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
        
    Returns:
        The directory path
    """
    os.makedirs(directory, exist_ok=True)
    return directory


def copy_files(src_pattern: str, dst_dir: str, recursive: bool = False) -> List[str]:
    """
    Copy files matching a pattern to destination directory.
    
    Args:
        src_pattern: Source file pattern or directory
        dst_dir: Destination directory
        recursive: Whether to copy recursively
        
    Returns:
        List of copied file paths
    """
    import glob
    
    ensure_dir(dst_dir)
    copied_files = []
    
    # Get source files
    if os.path.isfile(src_pattern):
        src_files = [src_pattern]
    elif os.path.isdir(src_pattern):
        if recursive:
            src_files = glob.glob(os.path.join(src_pattern, "**", "*"), recursive=True)
            src_files = [f for f in src_files if os.path.isfile(f)]
        else:
            src_files = glob.glob(os.path.join(src_pattern, "*"))
            src_files = [f for f in src_files if os.path.isfile(f)]
    else:
        src_files = glob.glob(src_pattern)
    
    # Copy files
    for src_file in src_files:
        if os.path.isfile(src_file):
            filename = os.path.basename(src_file)
            dst_file = os.path.join(dst_dir, filename)
            shutil.copy2(src_file, dst_file)
            copied_files.append(dst_file)
    
    return copied_files


def download_file(url: str, 
                 output_path: str, 
                 chunk_size: int = 8192,
                 show_progress: bool = True) -> bool:
    """
    Download a file from URL with progress bar.
    
    Args:
        url: URL to download from
        output_path: Local path to save file
        chunk_size: Download chunk size in bytes
        show_progress: Whether to show progress bar
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        ensure_dir(os.path.dirname(output_path))
        
        # Get file size for progress bar
        with urllib.request.urlopen(url) as response:
            total_size = int(response.info().get('Content-Length', -1))
        
        # Download with progress bar
        with urllib.request.urlopen(url) as response:
            with open(output_path, 'wb') as f:
                if show_progress and total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            pbar.update(len(chunk))
                else:
                    shutil.copyfileobj(response, f)
        
        print(f"Downloaded: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes
    """
    return os.path.getsize(file_path) if os.path.exists(file_path) else 0


def get_directory_size(directory: str) -> int:
    """
    Get total size of directory in bytes.
    
    Args:
        directory: Directory path
        
    Returns:
        Total size in bytes
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += get_file_size(file_path)
    return total_size


def clean_directory(directory: str, keep_patterns: Optional[List[str]] = None) -> int:
    """
    Clean directory, optionally keeping files matching patterns.
    
    Args:
        directory: Directory to clean
        keep_patterns: List of patterns to keep (glob patterns)
        
    Returns:
        Number of files removed
    """
    import glob
    
    if not os.path.exists(directory):
        return 0
    
    removed_count = 0
    keep_patterns = keep_patterns or []
    
    # Get all files to keep
    keep_files = set()
    for pattern in keep_patterns:
        keep_files.update(glob.glob(os.path.join(directory, pattern)))
    
    # Remove files not in keep list
    for file_path in glob.glob(os.path.join(directory, "*")):
        if os.path.isfile(file_path) and file_path not in keep_files:
            os.remove(file_path)
            removed_count += 1
    
    return removed_count


def create_directory_structure(base_dir: str, structure: dict) -> List[str]:
    """
    Create directory structure from nested dictionary.
    
    Args:
        base_dir: Base directory path
        structure: Nested dictionary defining structure
        
    Returns:
        List of created directories
    """
    created_dirs = []
    
    def _create_recursive(current_dir: str, current_structure: dict):
        for name, content in current_structure.items():
            new_dir = os.path.join(current_dir, name)
            ensure_dir(new_dir)
            created_dirs.append(new_dir)
            
            if isinstance(content, dict):
                _create_recursive(new_dir, content)
    
    ensure_dir(base_dir)
    created_dirs.append(base_dir)
    
    if isinstance(structure, dict):
        _create_recursive(base_dir, structure)
    
    return created_dirs


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes value as human readable string.
    
    Args:
        bytes_value: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"