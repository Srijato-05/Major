# Project Completion Roadmap: Adaptive Multi-Modal AI E-Waste Sorting Simulation

This roadmap documents the technical audit of the current codebase and defines the remaining phases required to bring the project to scientific rigor, physical simulation fidelity, and full UI/runtime completeness.

---

## 1. Current State & Audit Summary

Although initial boilerplate, a DeepLabV3-ResNet50 checkpoint (`models/checkpoints/multimodal_4ch_latest.pth`), and a FastAPI dashboard exist, several critical gaps remain:

- **Dataset Coverage & Proxy Mapping**:
  - **SpectralWaste (`data/raw/spectralwaste/`)**: Only the 852 labeled PNG triplets were fetched (omitting multi-GB 3D hyperspectral cubes and unlabeled scenes), and foreground pixels are heavily dominated by polythene films/bags (`BFR Polymers`). Other WEEE classes (`IC Molds`, `Metallic Heat Sinks`, `Copper Wire Harnesses`) rely on SpectralWaste stand-in proxies (`cardboard`, `basket`, `video_tape`, `filament`).
  - **Battery Packs (`data/raw/battery_packs/`)**: 944 `.jpg` images from `kdkd1/waste-garbage-management-dataset` with Otsu-synthesized masks and a synthetic NIR channel (`scripts/ingest_battery_dataset.py`) that adds `+60` brightness directly onto masked pixels—leaking the ground-truth mask into Channel 4.
  - **DeepPCB (`data/raw/deeppcb/`)**: Only 5 sample pairs are present and unused.
- **Class 0 Collision (Background vs. High-Grade PCB)**:
  - `configs/system_config.yaml` and `src/sorting/advanced_decision_engine.py` define Class 0 as `High-Grade PCB`, while dataset mappings assign `0 = Background` and `src/models/inference_engine.py` skips `c == 0`. Consequently, High-Grade PCBs are never detected or sorted.
- **No Train / Validation / Test Split**:
  - Training scripts (including `scripts/train_complete_corpus.py`) evaluate on the exact same 1,796 images they train on (only toggling augmentations off). `data/processed/` is empty, leaving no held-out generalization metrics or overfitting diagnostics.
- **Resolution & Minority-Class Underfitting**:
  - Training runs at `320×320` while inference runs at `640×640`. Minority classes (notably `Copper Wire Harnesses` at 32.76% in-sample IoU) underfit severely.
- **Unwired UI & Non-Continuous Simulation**:
  - `src/simulation/digital_twin.py` streams unrelated static images sequentially (852 SpectralWaste frames followed by 944 Battery images) rather than a continuous scrolling conveyor belt, preventing true multi-frame ByteTrack tracking and speed-slider physics.
  - The confidence slider updates `conf_threshold` in `app.py`, which `inference_engine.py` ignores in favor of hardcoded thresholds (`0.58` / `0.45`).
  - `/api/upload` advertises `.mp4` support in `index.html` but only supports single images (`cv2.imdecode`).
  - `bin_purity_scores` in `src/telemetry/advanced_telemetry_engine.py` is unpopulated, live charts are missing, and `index.html` hardcodes the GPU badge and partial class accuracy readouts.
- **Duplicate/Prototype Codebase (`backend/`)**:
  - A parallel prototype lives in `backend/` with stubs and its own `backend/tests/__init__.py`, which shadows root `tests/` when running bare `pytest`.

---

## 2. Roadmap Phases & Deliverables

### Phase 0 — Architecture Freeze & Repo Hygiene
*Goal: Eliminate structural contradictions before modifying data or retraining.*

- [x] **Fix Taxonomy & Class 0 Collision (6 → 7 classes)**:
  - Strictly reserve `Class 0 = Conveyor / Background`.
  - Assign foreground WEEE classes `1–6`: `1: High-Grade PCB`, `2: IC Molds & Chips`, `3: Battery Pack Units`, `4: BFR Polymers`, `5: Metallic Heat Sinks`, `6: Copper Wire Harnesses`.
  - Update `configs/system_config.yaml`, `src/models/multimodal_segmenter.py`, `src/models/inference_engine.py`, `src/models/evaluation_suite.py`, and `src/sorting/advanced_decision_engine.py`.
- [x] **Resolve `backend/` vs. `src/` & Test Hygiene**:
  - Archive or remove the disconnected `backend/` prototype (after porting actuator/FSM logic in Phase 3) so bare `pytest` runs cleanly from the project root without `backend/tests/__init__.py` collisions.
  - Sync `requirements.txt` (add `python-multipart`, `huggingface_hub`, `tqdm`) and remove stale YOLO architecture strings from `configs/system_config.yaml` and `README.md`.

---

### Phase 1 — Real Multi-Dataset Ingestion & Clean Splits
*Goal: Replace proxy/leaked data with real, properly split multimodal datasets.*

- [x] **Full Real Dataset Ingestion (Disk-Managed / Streamed for GB-scale corpora)**:
  - **PCBs & ICs**: Added `scripts/ingest_deeppcb.py` to download/extract full **DeepPCB** (1,500 pairs) into `data/raw/deeppcb/` and generate Class 1 (`High-Grade PCB`) masks + grayscale NIR (processed 10 local samples via `--local-only`; run without `--local-only` with network access to pull the full archive).
  - **BFR Polymers**: Retained SpectralWaste (`data/raw/spectralwaste/`, 852 triplets) RGB + SWIR triplets mapped to the 7-class taxonomy.
  - **Batteries & Modality Dropout**: Updated `scripts/ingest_battery_dataset.py` to Class 3 and **removed the `+60` mask-leaking NIR boost** across all 944 local battery triplets (`--rebuild-local`). Added `spectral_dropout_prob` in `ManifestMultimodalDataset` for RGB-only generalization.
- [x] **Deterministic Stratified Splits (`data/processed/`)**:
  - Generated and froze `train.json` (1,264), `val.json` (272), `test.json` (270), and `split_summary.json` via `scripts/build_dataset_splits.py` (70/15/15 stratified by source, `seed=42`, zero overlap).
  - Kept `test.json` strictly untouched until final benchmarking.

---

### Phase 2 — Rigorous Training, Overfitting Control & Honest Benchmarking
*Goal: Train all 6 foreground classes properly and verify generalization.*

- [x] **Hardware-Optimized Training Pipeline (GTX 1650 4 GB VRAM)**:
  - Enabled PyTorch AMP (FP16) and gradient accumulation in `src/models/train_pipeline.py` (`MultimodalTrainer`) and aligned `src/models/inference_engine.py` with `CONFIG["model"]["img_size"]`.
  - Added multi-class Focal Cross-Entropy (`focal_gamma`) + Dice + inverse class weights in `src/models/composite_loss.py` and `scripts/train_rigorous_7class.py`.
  - Implemented Stage 1 (frozen backbone) → Stage 2 (full unfreeze, AdamW, Cosine Annealing, and early stopping on `val_miou`).
- [x] **Overfitting & Underfitting Diagnostics**:
  - Logged `train_loss`, `val_loss`, `val_miou`, `val_mf1`, `val_pixel_acc`, and per-class Val metrics to `models/checkpoints/training_history.json` and `training_history.csv`, saving `multimodal_4ch_best.pth` on best validation mIoU.
- [x] **Final Held-Out Test Benchmark Pipeline**:
  - Built `scripts/evaluate_heldout_test.py` to evaluate the untouched `data/processed/test.json` split with per-class Precision/Recall/IoU/F1, 7×7 confusion matrix, and 3-channel RGB vs. 4-channel RGB+SWIR ablation (`models/checkpoints/test_benchmark_report.json`). *(Verified end-to-end via smoke check; ready for full-corpus training run: `python scripts/train_rigorous_7class.py` followed by `python scripts/evaluate_heldout_test.py`.)*

---

### Phase 3 — Physically Realistic Digital Twin & Pneumatic Actuation
*Goal: Make tracking, kinematics, sorting, and purity physically meaningful.*

- [x] **Continuous Scrolling Conveyor Stream (`src/simulation/`)**:
  - Built `ScrollingConveyorStream` (`src/simulation/conveyor_stream.py`) compositing unseen `test.json` e-waste cutouts across all 6 foreground classes onto a moving belt canvas at `v_belt` (`1.0–4.0 m/s`), wired into `ConveyorDigitalTwinSimulator` (`src/simulation/digital_twin.py`).
  - Aligned the belt travel axis (`x-positive`) across `configs/conveyor_config.yaml` and `src/kinematics/conveyor_motion.py`.
- [x] **Pneumatic Nozzle Scheduler & Item Lifecycle FSM in `src/`**:
  - Built `PneumaticActuatorScheduler` & `ItemLifecycleRecord` FSM (`src/sorting/pneumatic_scheduler.py`) wired to the 64-nozzle manifold (`actuation_lag_ms`, `solenoid_pitch_mm`, `ejection_line_x_mm`).
- [x] **True Bin Purity % & Recovery Rate %**:
  - Added `record_sorting_accuracy()` in `src/telemetry/advanced_telemetry_engine.py`, populating `bin_purity_scores`, `class_recovery_rates`, and `overall_purity_pct`.

---

### Phase 4 — Dashboard Completion & Real-Time Optimization
*Goal: Wire all UI controls, add live analytics, and target `< 20 ms` (`≥ 50 FPS`) inference.*

- [ ] **Wire Broken / Missing UI Controls**:
  - Connect the **Detection Confidence slider** to `src/models/inference_engine.py` (removing hardcoded `0.58` / `0.45` thresholds).
  - Connect the **Belt Speed slider** to the continuous conveyor scroll step rate.
  - Implement `.mp4` video upload processing in `/api/upload` (`src/ui/app.py`).
  - Replace the hardcoded GPU badge in `src/ui/templates/index.html` with live device status and expand the accuracy panel to show all foreground classes (including Battery Annex VII Recall/F1).
- [ ] **Live Analytics Charts**:
  - Add real-time charts for throughput (`kg/hr`), per-bin purity/recovery (`%`), economic yield (`€/hr`), and pipeline latency (`ms`).
- [ ] **Inference Acceleration & Export**:
  - Implement FP16 inference and ONNX Runtime export/benchmarking to measure speedup toward the `< 20 ms/frame` real-time target.

---

### Phase 5 — End-to-End Verification & Documentation
*Goal: Ensure tests, benchmarks, and docs match reality.*

- [ ] **Expanded Test Suite**:
  - Add tests covering the 7-class taxonomy, zero-leakage train/val/test split integrity, pneumatic nozzle timing, `.mp4` upload handling, and confidence thresholding; ensure `pytest` passes from root.
- [ ] **Final Documentation Sync**:
  - Update `README.md` with honest held-out Test-set metrics, RGB vs. RGB+SWIR ablation results, and accurate model/architecture descriptions.

