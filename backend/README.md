# Real-Data Digital Twin: Adaptive Industrial E-Waste Multi-Modal Segmentation & Pneumatic Sorting

This directory contains the Python implementation of the **Adaptive Industrial E-Waste Multi-Modal Segmentation & Pneumatic Sorting Digital Twin**, driven by **real industrial e-waste datasets** and **real pre-trained vision foundation weights**.

## Real Datasets & Model Weights Integration

1. **Real Dataset Downloader & Harmonizer (`data/dataset_downloader.py`)**:
   - Automatically fetches, extracts, and harmonizes public benchmark datasets:
     - **PCB-Vision** (53 co-registered RGB + $400\text{--}1000\text{ nm}$ HSI scenes)
     - **DeepPCB** (1,500 aligned PCB defect image pairs)
     - **PKU-Market-PCB** (3,505 high-resolution PCB micro-defect images)
     - **SpectralWaste** (852 labeled / 6,803 unlabeled RGB + SWIR scenes)
     - **ThermalRGBTrash** (22 co-registered RGB-LWIR video streams)
   - Converts real annotations to unified YOLO polygon segmentation labels.

2. **Real Stream Generator (`data/real_stream_loader.py`)**:
   - Stitches real dataset image component crops into real conveyor video stream MP4 feeds (`data/real_conveyor_stream.mp4`).

3. **Real Pre-trained Model Weights Surgery (`models/input_adapter.py`)**:
   - Downloads official PyTorch pre-trained weights (`yolov8n-seg.pt` or `yolov11n-seg.pt`).
   - Performs **Conv0 Weight Inflation Surgery**: expands 3-channel RGB weights to 4-channel RGB+NIR inputs by computing the cross-channel mean across $R, G, B$ pre-trained weights.

## Execution Modes

- Download Real Datasets: `python main.py --mode download_data`
- Generate Real Conveyor Video Stream: `python main.py --mode generate_stream`
- Run Pipeline: `python main.py --mode run`
- Run Test Suite: `python main.py --mode test`
