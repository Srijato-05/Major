import os
import time
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from src.models.inference_engine import MultimodalInferenceEngine
from src.tracking.bytetrack import ByteTracker
from src.kinematics.conveyor_motion import ConveyorKinematicsEngine
from src.sorting.advanced_decision_engine import AdvancedSortingDecisionEngine, AdvancedSortedItemResult
from src.sorting.pneumatic_scheduler import PneumaticActuatorScheduler
from src.telemetry.advanced_telemetry_engine import AdvancedTelemetryEngine
from src.simulation.conveyor_stream import ScrollingConveyorStream
from src.data.multimodal_dataset import CombinedEWasteDataset
from src.utils.config_loader import CONFIG

class ConveyorDigitalTwinSimulator:
    """
    Complete Real-Time Industrial Conveyor Digital Twin Simulator.
    Orchestrates the entire multi-modal pipeline:
    1. Continuous Scrolling Conveyor Stream (from unseen test.json items across all 6 WEEE classes)
    2. Deep 4-Channel AI Segmentation & Mask Extraction (FP16-accelerated on CUDA)
    3. ReID-Free ByteTrack Object Tracking across belt motion
    4. Homography Coordinate Transform (Pixels -> Millimeters)
    5. Advanced Sorting Decision Evaluation & 64-Nozzle Pneumatic Actuation Scheduler
    6. Industrial Telemetry Accounting (Throughput, Bin Purity %, Recovery %, P&L €/hr)
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        belt_speed_mps: Optional[float] = None
    ):
        if weights_path is None:
            best_chk = "models/checkpoints/multimodal_4ch_best.pth"
            latest_chk = "models/checkpoints/multimodal_4ch_latest.pth"
            weights_path = best_chk if os.path.exists(best_chk) else latest_chk
        self.kinematics = ConveyorKinematicsEngine(speed_mps=belt_speed_mps)
        self.inference_engine = MultimodalInferenceEngine(weights_path=weights_path)
        self.tracker = ByteTracker()
        self.decision_engine = AdvancedSortingDecisionEngine()
        self.pneumatic_scheduler = PneumaticActuatorScheduler(speed_mps=self.kinematics.speed_mps)
        self.telemetry = AdvancedTelemetryEngine()
        from src.models.evaluation_suite import ModelEvaluationSuite
        num_classes = CONFIG.get("model", {}).get("num_classes", 7)
        self.evaluator = ModelEvaluationSuite(num_classes=num_classes)

        # Track history to prevent double-sorting the same item ID
        self.evaluated_track_ids = set()
        from collections import deque
        self.sorted_history = deque(maxlen=30)

        # Use continuous scrolling conveyor stream when test.json exists; fallback to static dataset
        self.stream = ScrollingConveyorStream(
            manifest_path="data/processed/test.json",
            speed_mps=self.kinematics.speed_mps,
        ) if os.path.exists("data/processed/test.json") else None

        self.dataset = None if self.stream is not None else CombinedEWasteDataset(
            spectral_dir="data/raw/spectralwaste",
            battery_dir="data/raw/battery_packs",
            is_training=False
        )
        self.current_frame_idx = 0

    def set_belt_speed(self, new_speed_mps: float):
        """Synchronizes conveyor speed across kinematics, scrolling stream, and pneumatic scheduler."""
        self.kinematics.set_speed(new_speed_mps)
        self.pneumatic_scheduler.speed_mps = self.kinematics.speed_mps
        if self.stream is not None:
            self.stream.set_speed(self.kinematics.speed_mps)

    def process_frame(
        self,
        rgb_frame: np.ndarray,
        spectral_band: Optional[np.ndarray] = None,
        gt_mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end processing on a single conveyor frame.
        """
        t_start = time.perf_counter()
        now_ns = time.monotonic_ns()

        # 1. AI Detection & Segmentation
        pred_res = self.inference_engine.predict_frame(rgb_frame, spectral_band)
        instances = pred_res.get("instances", [])

        # 2. Multi-Object Tracking
        active_tracklets = self.tracker.update(instances)

        # 3. Conveyor Kinematics, Sorting Decision & 64-Nozzle Pneumatic Scheduling
        current_frame_items: List[Dict[str, Any]] = []
        newly_sorted: List[AdvancedSortedItemResult] = []

        for tracklet in active_tracklets:
            metric_info = self.kinematics.project_tracklet_metric(tracklet)
            xw, yw = metric_info["metric_pos_mm"]

            if tracklet.track_id not in self.evaluated_track_ids:
                decision_res = self.decision_engine.evaluate_item(
                    track_id=tracklet.track_id,
                    class_id=tracklet.class_id,
                    confidence=tracklet.score,
                    area_mm2=metric_info["metric_area_mm2"]
                )
                self.telemetry.record_item(decision_res)

                if gt_mask is not None:
                    tlwh = tracklet.to_tlwh()
                    gx1 = max(0, int(tlwh[0]))
                    gy1 = max(0, int(tlwh[1]))
                    gx2 = min(gt_mask.shape[1], int(tlwh[0] + tlwh[2]))
                    gy2 = min(gt_mask.shape[0], int(tlwh[1] + tlwh[3]))
                    if gx2 > gx1 and gy2 > gy1:
                        roi = gt_mask[gy1:gy2, gx1:gx2]
                        fg_vals = roi[roi > 0]
                        gt_cid = int(np.bincount(fg_vals).argmax()) if fg_vals.size > 0 else 0
                        self.telemetry.record_sorting_accuracy(
                            predicted_bin=decision_res.target_bin,
                            predicted_class_id=tracklet.class_id,
                            gt_class_id=gt_cid,
                        )

                self.evaluated_track_ids.add(tracklet.track_id)
                newly_sorted.append(decision_res)
                self.sorted_history.appendleft(decision_res.__dict__)
                target_bin = decision_res.target_bin
            else:
                target_bin = self.decision_engine.taxonomy[tracklet.class_id].target_bin

            self.pneumatic_scheduler.update_and_schedule(
                track_id=tracklet.track_id,
                class_id=tracklet.class_id,
                x_w_mm=xw,
                y_w_mm=yw,
                target_bin=target_bin,
                speed_mps=self.kinematics.speed_mps,
                now_ns=now_ns,
            )

            current_frame_items.append(metric_info)

        due_pulses = self.pneumatic_scheduler.pop_due_pulses(now_ns=now_ns)

        # 4. Record Latency
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        self.telemetry.record_latency(latency_ms)

        # 5. Visual Rendering on RGB frame
        annotated_canvas = self.inference_engine.annotate_frame(rgb_frame, pred_res)

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
            "pneumatic_pulses": [p.__dict__ for p in due_pulses],
            "pneumatic_stats": self.pneumatic_scheduler.get_stats(),
            "telemetry": self.telemetry.get_dashboard_metrics(),
            "accuracy_metrics": self.evaluator.compute_metrics(),
            "latency_ms": round(latency_ms, 2)
        }

    def step_simulation(self) -> Dict[str, Any]:
        """
        Advances the simulation by 1 frame (continuous scrolling conveyor stream or dataset fallback)
        and updates dynamic accuracy metrics against ground-truth masks.
        """
        if self.stream is not None:
            rgb, hsi, gt_mask = self.stream.step()
        elif self.dataset is not None and len(self.dataset) > 0:
            sample = self.dataset[self.current_frame_idx % len(self.dataset)]
            self.current_frame_idx += 1
            tensor_img = sample["image"].permute(1, 2, 0).numpy() * 255.0
            rgb = tensor_img[:, :, :3].astype(np.uint8)
            hsi = tensor_img[:, :, 3].astype(np.uint8)
            gt_mask = sample["mask"].numpy()
        else:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            return self.process_frame(dummy)

        res = self.process_frame(rgb, hsi, gt_mask=gt_mask)

        # Update dynamic accuracy metrics against ground truth
        pred_mask = res["raw_predictions"].get("mask")
        if pred_mask is not None:
            self.evaluator.update(pred_mask, gt_mask)
            res["accuracy_metrics"] = self.evaluator.compute_metrics()

        return res
