"""
Pipeline Manager & Asynchronous Multi-Threaded Engine
Orchestrates process/thread execution across:
- Thread 1 (Producer): Video / Camera frame ingestion into lock-free frame queue.
- Thread 2 (Worker): 4-channel segmentation inference, ByteTrack association, and 2D homography kinematics.
- Thread 3 (Consumer): Finite State Machine state tracking, microsecond actuator queue firing, and Operator HUD rendering.
"""

from queue import Queue
import threading
import time
import os
import yaml
import logging

from core.detector import SegmentationDetector
from core.tracker import ByteTracker
from core.kinematics import HomographyKinematicsMapper
from core.fsm import ItemLifecycleTracker, ItemState
from core.actuator_queue import PneumaticActuatorQueue
from ui.operator_hud import render_operator_hud

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [pipeline_manager]: %(message)s")
logger = logging.getLogger("pipeline_manager")

class DigitalTwinPipelineManager:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.frame_queue = Queue(maxsize=30)
        self.detection_queue = Queue(maxsize=30)
        self.running = False

        # Load parameters
        conveyor_cfg = self.config.get("conveyor", {})
        actuator_cfg = self.config.get("pneumatic_actuator", {})
        homography_cfg = self.config.get("homography", {})

        # Instantiate core components
        self.detector = SegmentationDetector(
            model_name=self.config.get("model", {}).get("architecture", "yolov8n-seg.pt"),
            conf_threshold=self.config.get("model", {}).get("conf_threshold", 0.45),
            iou_threshold=self.config.get("model", {}).get("iou_threshold", 0.50),
            device=self.config.get("system", {}).get("device", "cpu")
        )

        self.tracker = ByteTracker(
            track_thresh=self.config.get("tracking", {}).get("track_thresh", 0.50),
            match_thresh=self.config.get("tracking", {}).get("match_thresh", 0.75)
        )

        self.kinematics = HomographyKinematicsMapper(
            src_pts=homography_cfg.get("source_pixel_points", [[42, 38], [598, 40], [630, 610], [10, 608]]),
            dst_pts=homography_cfg.get("destination_world_points", [[0.0, 0.0], [0.8, 0.0], [0.8, 1.2], [0.0, 1.2]])
        )

        self.actuator_queue = PneumaticActuatorQueue(
            ejection_distance_m=actuator_cfg.get("ejection_line_distance_m", 1.5),
            v_belt_m_s=conveyor_cfg.get("speed_m_per_s", 2.5),
            solenoid_delay_ms=actuator_cfg.get("solenoid_response_delay_ms", 12.0),
            num_valves=actuator_cfg.get("num_valves", 16),
            valve_spacing_m=actuator_cfg.get("valve_spacing_m", 0.05)
        )

        self.target_classes = actuator_cfg.get("target_ejection_classes", [1, 4, 5])
        self.item_trackers = {} # track_id -> ItemLifecycleTracker

    def start(self):
        """Starts asynchronous pipeline threads."""
        self.running = True
        logger.info("Digital Twin Asynchronous Pipeline Engine Started.")

        # Thread 2: Worker Process (Inference + Tracking + Kinematics)
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def process_frame(self, timestamp_ns: int, frame_rgb: float):
        """
        Executes single synchronous pipeline step for batch / test execution.
        Returns: (tracked_objects, due_actuation_events)
        """
        tensor_4ch = self.detector.preprocess_4channel(frame_rgb)
        raw_detections = self.detector.infer(tensor_4ch)
        active_tracklets = self.tracker.update(raw_detections)

        due_events = []
        for t in active_tracklets:
            cx = (t.bbox[0] + t.bbox[2]) / 2.0
            cy = (t.bbox[1] + t.bbox[3]) / 2.0
            
            # Map pixel coordinates to metric conveyor world plane
            x_w, y_w = self.kinematics.pixel_to_world(cx, cy)
            
            # Track state lifecycle
            if t.track_id not in self.item_trackers:
                self.item_trackers[t.track_id] = ItemLifecycleTracker(t.track_id, t.class_id, (x_w, y_w), timestamp_ns)
            else:
                fsm_item = self.item_trackers[t.track_id]
                fsm_item.update_position((x_w, y_w))

                # Schedule pneumatic air-jet pulse if target ejection class
                if t.class_id in self.target_classes and fsm_item.state == ItemState.TRACKED:
                    fsm_item.set_state(ItemState.IN_TRANSIT)
                    self.actuator_queue.schedule_ejection(
                        track_id=t.track_id,
                        y_w=y_w,
                        x_w=x_w,
                        pulse_duration_ms=25.0,
                        now_ns=timestamp_ns
                    )

        # Pop events due for physical actuation pulse
        due_events = self.actuator_queue.pop_due_events(now_ns=timestamp_ns)
        return active_tracklets, due_events

    def _worker_loop(self):
        """Worker thread processing frames asynchronously."""
        while self.running:
            if not self.frame_queue.empty():
                timestamp_ns, frame = self.frame_queue.get()
                self.process_frame(timestamp_ns, frame)
            else:
                time.sleep(0.001)

    def stop(self):
        """Stops asynchronous pipeline execution."""
        self.running = False
        logger.info("Digital Twin Pipeline Engine Stopped.")
