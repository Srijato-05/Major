import time
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Generator
from src.models.inference_engine import MultimodalInferenceEngine
from src.tracking.bytetrack import ByteTracker, STrack
from src.kinematics.conveyor_motion import ConveyorKinematicsEngine
from src.sorting.advanced_decision_engine import AdvancedSortingDecisionEngine, AdvancedSortedItemResult
from src.telemetry.advanced_telemetry_engine import AdvancedTelemetryEngine
from src.data.multimodal_dataset import SpectralWasteDataset
from src.utils.config_loader import CONFIG

class ConveyorDigitalTwinSimulator:
    """
    Complete Real-Time Industrial Conveyor Digital Twin Simulator.
    Orchestrates the entire multi-modal pipeline:
    1. Continuous Conveyor Frame Streaming (from benchmark dataset or uploaded video/images)
    2. Deep 4-Channel AI Segmentation & Mask Extraction
    3. ReID-Free ByteTrack Object Tracking across line movement
    4. Homography Coordinate Transform (Pixels -> Millimeters)
    5. Advanced Sorting Decision Evaluation (Annex VII & EPR compliance)
    6. Industrial Telemetry Accounting (Throughput, Purity, P&L in €/hr)
    """

    def __init__(
        self,
        weights_path: Optional[str] = "models/checkpoints/multimodal_4ch_latest.pth",
        belt_speed_mps: Optional[float] = None
    ):
        self.kinematics = ConveyorKinematicsEngine(speed_mps=belt_speed_mps)
        self.inference_engine = MultimodalInferenceEngine(weights_path=weights_path)
        self.tracker = ByteTracker()
        self.decision_engine = AdvancedSortingDecisionEngine()
        self.telemetry = AdvancedTelemetryEngine()
        from src.models.evaluation_suite import ModelEvaluationSuite
        self.evaluator = ModelEvaluationSuite(num_classes=6)

        # Track history to prevent double-sorting the same item ID
        self.evaluated_track_ids = set()
        from collections import deque
        self.sorted_history = deque(maxlen=30)

        # Combined dataset stream iterator: 852 SpectralWaste + 944 Battery Pack units
        from src.data.multimodal_dataset import CombinedEWasteDataset
        self.dataset = CombinedEWasteDataset(
            spectral_dir="data/raw/spectralwaste",
            battery_dir="data/raw/battery_packs",
            is_training=False
        )
        self.current_frame_idx = 0

    def process_frame(
        self,
        rgb_frame: np.ndarray,
        spectral_band: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end processing on a single conveyor frame.
        """
        t_start = time.perf_counter()

        # 1. AI Detection & Segmentation
        pred_res = self.inference_engine.predict_frame(rgb_frame, spectral_band)
        instances = pred_res.get("instances", [])

        # 2. Multi-Object Tracking
        active_tracklets = self.tracker.update(instances)

        # 3. Conveyor Kinematics & Sorting Decision
        current_frame_items: List[Dict[str, Any]] = []
        newly_sorted: List[AdvancedSortedItemResult] = []

        for tracklet in active_tracklets:
            # Metric projection via Homography
            metric_info = self.kinematics.project_tracklet_metric(tracklet)

            # Evaluate decision rules
            if tracklet.track_id not in self.evaluated_track_ids:
                decision_res = self.decision_engine.evaluate_item(
                    track_id=tracklet.track_id,
                    class_id=tracklet.class_id,
                    confidence=tracklet.score,
                    area_mm2=metric_info["metric_area_mm2"]
                )
                self.telemetry.record_item(decision_res)
                self.evaluated_track_ids.add(tracklet.track_id)
                newly_sorted.append(decision_res)
                self.sorted_history.appendleft(decision_res.__dict__)

            current_frame_items.append(metric_info)

        # 4. Record Latency
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        self.telemetry.record_latency(latency_ms)

        # 5. Visual Rendering on RGB frame
        annotated_canvas = self.inference_engine.annotate_frame(rgb_frame, pred_res)

        # Draw persistent tracklet badges & metric positions
        for item in current_frame_items:
            u, v = item["pixel_center"]
            tid = item["track_id"]
            xw, yw = item["metric_pos_mm"]
            badge_text = f"ID #{tid} | ({xw:.0f}, {yw:.0f} mm)"
            cv2.putText(
                annotated_canvas,
                badge_text,
                (int(u) - 30, int(v) + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return {
            "annotated_frame": annotated_canvas,
            "raw_predictions": pred_res,
            "active_tracklets": current_frame_items,
            "newly_sorted": [r.__dict__ for r in newly_sorted],
            "sorted_history": list(self.sorted_history),
            "telemetry": self.telemetry.get_dashboard_metrics(),
            "accuracy_metrics": self.evaluator.compute_metrics(),
            "latency_ms": round(latency_ms, 2)
        }

    def step_simulation(self) -> Dict[str, Any]:
        """
        Advances the simulation by 1 frame from the real benchmark dataset
        and updates dynamic accuracy metrics against ground-truth masks.
        """
        if len(self.dataset) == 0:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            return self.process_frame(dummy)

        sample = self.dataset[self.current_frame_idx % len(self.dataset)]
        self.current_frame_idx += 1

        # Tensor [4, H, W] -> RGB (H, W, 3) and HSI (H, W)
        tensor_img = sample["image"].permute(1, 2, 0).numpy() * 255.0
        rgb = tensor_img[:, :, :3].astype(np.uint8)
        hsi = tensor_img[:, :, 3].astype(np.uint8)
        gt_mask = sample["mask"].numpy()

        res = self.process_frame(rgb, hsi)

        # Update dynamic accuracy metrics against ground truth
        pred_mask = res["raw_predictions"].get("mask")
        if pred_mask is not None:
            self.evaluator.update(pred_mask, gt_mask)
            res["accuracy_metrics"] = self.evaluator.compute_metrics()

        return res
