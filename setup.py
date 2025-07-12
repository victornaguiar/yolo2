"""Setup configuration for the soccer tracking pipeline."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="soccer-tracking-pipeline",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive computer vision pipeline for tracking soccer players using YOLOv8 and various tracking algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/soccer-tracking-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords="computer-vision, object-tracking, yolo, soccer, multi-object-tracking",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/soccer-tracking-pipeline/issues",
        "Source": "https://github.com/yourusername/soccer-tracking-pipeline",
    },
)