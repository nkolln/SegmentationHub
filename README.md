# Paintable Surface Segmentation

## Overview
This project implements a semantic segmentation model using Segformer architecture to identify paintable surfaces on building walls.

## Project Structure
- `configs/`: Configuration files.
- `notebooks/`: Jupyter notebooks for exploration.
- `src/`: Source code.
  - `data/`: Dataset and transforms.
  - `models/`: Model architectures (Segformer, U-Net).
  - `training/`: Training loop and losses.
  - `evaluation/`: Metrics and visualization.
  - `utils/`: Logging and helper functions.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Train the model:
```bash
python train.py --config configs/config.yaml
```
