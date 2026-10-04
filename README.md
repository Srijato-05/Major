# Adaptive Multi-Modal AI E-Waste Sorting Simulation

An end-to-end, real-time software digital twin simulating an industrial optical sorting system for Waste Electrical and Electronic Equipment (WEEE). Built using real multimodal datasets, pretrained vision backbones adapted via 4-channel tensor inflation, ByteTrack multi-object tracking, and planar metric homography.

## Project Structure

```
Major/
├── configs/
│   └── system_config.yaml           # Central kinematics, taxonomy, and system parameters
├── data/
│   ├── raw/                         # Raw multimodal benchmark datasets
│   ├── processed/                   # Processed 4-channel tensors
│   └── sample_streams/              # Simulation sequence feeds
├── docs/
│   ├── doc-1.pdf                    # Technical Architecture specification
│   ├── doc-2.pdf                    # Multimodal Sensor Fusion & ML benchmarks
│   └── project_plan/
│       └── IMPLEMENTATION_PLAN.md   # Complete project roadmap & phase breakdown
├── src/
│   ├── data/                        # Dataset loaders, augmentations & 4-channel builder
│   ├── models/
│   │   └── weight_inflation.py      # Conv1 4-channel tensor inflation (mean RGB preservation)
│   ├── tracking/                    # ByteTrack MOT implementation
│   ├── kinematics/
│   │   └── homography.py            # 2D Planar Homography (H) mapping pixels to conveyor mm
│   ├── sorting/
│   │   └── decision_engine.py       # Sorting logic, mass estimation, & Annex VII bin routing
│   ├── telemetry/
│   │   └── telemetry_engine.py      # Industrial KPIs (throughput, purity, net €/hr, latency)
│   ├── simulation/                  # Conveyor belt & stream generator
│   ├── ui/                          # High-performance simulation dashboard
│   └── utils/
│       └── config_loader.py         # Config reader singleton
├── tests/
│   └── test_foundation.py           # PyTest test suite
├── requirements.txt                 # Pinned dependencies
└── README.md
```

## Setup & Verification

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run unit tests
pytest tests/ -v
```
