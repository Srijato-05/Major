import pytest
import numpy as np
from src.tracking.kalman_filter import ConveyorKalmanFilter
from src.tracking.bytetrack import ByteTracker, STrack
from src.kinematics.conveyor_motion import ConveyorKinematicsEngine

def test_kalman_filter_init_and_predict():
    kf = ConveyorKalmanFilter()
    measurement = np.array([100.0, 200.0, 1.5, 40.0])  # x, y, a, h
    mean, cov = kf.initiate(measurement)
    assert mean.shape == (8,)
    assert cov.shape == (8, 8)

    # Predict step
    mean_p, cov_p = kf.predict(mean, cov)
    assert mean_p.shape == (8,)
    assert cov_p.shape == (8, 8)

def test_bytetrack_association_and_persistence():
    tracker = ByteTracker(track_thresh=0.5)

    # Frame 1: 2 items appear
    dets_frame1 = [
        {"bbox": [100, 100, 50, 50], "confidence": 0.9, "class_id": 0, "class_name": "High-Grade PCB", "polygon": []},
        {"bbox": [300, 200, 40, 40], "confidence": 0.85, "class_id": 1, "class_name": "IC Molds & Chips", "polygon": []}
    ]
    tracks1 = tracker.update(dets_frame1)
    assert len(tracks1) == 2
    id1, id2 = tracks1[0].track_id, tracks1[1].track_id
    assert id1 != id2

    # Frame 2: Items shift slightly downstream (simulating belt motion)
    dets_frame2 = [
        {"bbox": [115, 100, 50, 50], "confidence": 0.88, "class_id": 0, "class_name": "High-Grade PCB", "polygon": []},
        {"bbox": [315, 200, 40, 40], "confidence": 0.82, "class_id": 1, "class_name": "IC Molds & Chips", "polygon": []}
    ]
    tracks2 = tracker.update(dets_frame2)
    assert len(tracks2) == 2

    # Verify ID persistence across frames (no identity switch)
    current_ids = {t.track_id for t in tracks2}
    assert id1 in current_ids
    assert id2 in current_ids

def test_bytetrack_low_confidence_recovery():
    # Stage 2 association: recover motion-blurred fragment with score 0.3
    tracker = ByteTracker(track_thresh=0.5)

    # Frame 1: Normal detection
    d1 = [{"bbox": [100, 100, 50, 50], "confidence": 0.9, "class_id": 0, "class_name": "High-Grade PCB"}]
    t1 = tracker.update(d1)
    orig_id = t1[0].track_id

    # Frame 2: Low-confidence detection due to blur (score = 0.35)
    d2 = [{"bbox": [108, 100, 50, 50], "confidence": 0.35, "class_id": 0, "class_name": "High-Grade PCB"}]
    t2 = tracker.update(d2)
    assert len(t2) == 1
    assert t2[0].track_id == orig_id  # Successfully recovered without creating new ID

def test_conveyor_kinematics_engine():
    kinematics = ConveyorKinematicsEngine(speed_mps=2.0)
    
    # Fake tracklet
    tracklet = STrack(tlwh=[100, 200, 50, 60], score=0.92, class_id=0, class_name="High-Grade PCB")
    tracklet.track_id = 42

    proj = kinematics.project_tracklet_metric(tracklet)
    assert proj["track_id"] == 42
    assert "metric_pos_mm" in proj
    assert "metric_area_mm2" in proj
    assert proj["belt_speed_mps"] == 2.0
    assert proj["time_to_ejection_sec"] > 0

    # Test dynamic speed adjustment
    kinematics.set_speed(3.5)
    assert kinematics.speed_mps == 3.5
