# COMP0248 3D Object Detection Coursework

This repository contains the implementation of three pipelines for table detection and segmentation using the Sun3D dataset.

## Project Structure

```
Code/
  ├── src/
  │    ├── data_utils/      # Common data processing utilities
  │    ├── pipelineA/       # Point cloud classification
  │    ├── pipelineB/       # Monocular depth estimation + classification
  │    ├── pipelineC/       # Point cloud segmentation
  │    └── utils/           # Common utilities
  ├── data/                 # Data directory
  │    └── CW2-Dataset/     # Sun3D dataset
  ├── results/              # Logs, prediction outputs, plots, etc.
  ├── weights/              # Model checkpoints
  ├── requirements.txt      # Python dependencies
  └── README.md             # This file
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install learning3d:
```bash
git clone https://github.com/vinits5/learning3d.git
cd learning3d
pip install -e .
cd ..
```

## Data Preparation

The Sun3D dataset should be placed in the `data/CW2-Dataset` directory with the following structure:
- Train data: mit_32_d507, mit_76_459, mit_76_studyroom, mit_gym_z_squash, mit_lab_hj
- Test data: harvard_c5, harvard_c6, harvard_c11, harvard_tea_2

## Running the Pipelines

### Pipeline A: Point Cloud Classification

```bash
python src/pipelineA/train.py
python src/pipelineA/eval.py
```

### Pipeline B: Monocular Depth Estimation + Classification

```bash
python src/pipelineB/train.py
python src/pipelineB/eval.py
```

### Pipeline C: Point Cloud Segmentation

```bash
python src/pipelineC/train.py
python src/pipelineC/eval.py
```

## Results

Evaluation results, visualization outputs, and model logs are saved in the `results` directory. 