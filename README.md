# README.md

# Prokudin-Gorskii Image Alignment

This project implements automatic alignment of Prokudin-Gorskii glass plate photographs using computer vision techniques.

## Overview

The Prokudin-Gorskii collection contains early color photographs taken by Sergei Mikhailovich Prokudin-Gorskii between 1907-1915. Each image consists of three separate exposures (Blue, Green, Red channels) that need to be aligned to create a single color photograph.

## Features

- **Single-scale alignment** for small JPEG images using exhaustive search
- **Multi-scale pyramid alignment** for large TIFF images using coarse-to-fine optimization
- **Edge-based features** using Sobel gradients for robust alignment
- **Normalized Cross-Correlation (NCC)** scoring metric
- **Bells & Whistles enhancements:**
  - Automatic border cropping
  - White balance correction
  - Automatic contrast adjustment

## Requirements

```bash
pip install Pillow scipy numpy
```

## Usage

### Basic Usage

```python
from main import ProkudinGorskiiAligner

# Create aligner instance
aligner = ProkudinGorskiiAligner()

# Process a single image
aligner.process("images/cathedral.jpg", "output/cathedral_restored.jpg")
```

### Batch Processing

```python
# Process all images in a directory
aligner.batch_process("images", "output")

# Process specific images
subset = ["cathedral.jpg", "monastery.jpg", "emir.tif"]
aligner.batch_process("images", "output", file_list=subset)
```

### Command Line

```bash
python main.py
```

## Algorithm Details

### Single-Scale Alignment (JPEGs)
- Used for small images (cathedral.jpg, monastery.jpg, tobolsk.jpg)
- Exhaustive search over displacement window [-15, 15] pixels
- Direct application of NCC metric on Sobel edge features

### Multi-Scale Pyramid Alignment (TIFFs)
- Used for large images (emir.tif, harvesters.tif, etc.)
- Recursive coarse-to-fine alignment across 5 pyramid levels
- 2x downsampling at each level using scipy.ndimage.zoom
- Refines alignment with ±2 pixel search at each level

### Edge Features
- Sobel gradient magnitude: `sqrt(dx² + dy²)` using scipy.ndimage.sobel
- Robust to brightness differences between channels
- Essential for problematic images like emir.tif

### Implementation
- Image I/O: PIL (Pillow) for reading and writing images
- Edge detection: scipy.ndimage.sobel for gradient computation
- Image resizing: scipy.ndimage.zoom for pyramid downsampling
- Normalization: numpy operations for 8-bit conversion

## File Structure

```
├── main.py              # Main implementation
├── images/               # Input images directory
│   ├── cathedral.jpg
│   ├── monastery.jpg
│   ├── emir.tif
│   └── ...
├── output/               # Generated output directory
├── index.html           # Results webpage
└── README.md           # This file
```

## Parameters

- `search_range=15`: Maximum displacement search window
- `pyramid_depth=5`: Number of pyramid levels for multi-scale alignment
- `crop_ratio=0.15`: Border cropping ratio to avoid edge artifacts

## Results

See `index.html` for visual results and computed alignment offsets for all processed images.

## Author

Mohak Sharma (ms7306)
