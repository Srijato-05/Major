import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from src.kinematics.homography import PlanarHomography
from src.utils.config_loader import CONFIG

class ConveyorKinematicsEngine:
    """
    Simulates and projects continuous conveyor motion in metric space (mm):
    - Transforms tracklet image coordinates (u, v) to conveyor coordinates (x_w, y_w) in mm.
    - Estimates true conveyor velocity vector (v_x, v_y) in m/s.
    - Projects arrival timestamps at the sorting threshold / pneumatic ejection line.
    """

    def __init__(
        self,
        speed_mps: Optional[float] = None,
        homography: Optional[PlanarHomography] = None
    ):
        cfg_conv = CONFIG.get("conveyor", {})
        cfg_geom = CONFIG.get("optical_geometry", {})

        self.speed_mps = speed_mps if speed_mps is not None else cfg_conv.get("speed_mps", 2.5)
        self.fps = cfg_conv.get("sampling_rate_fps", 60)

        # Initialize homography if not provided
        if homography is not None:
            self.homography = homography
        else:
            scale_x = cfg_geom.get("scale_x_mm_per_px", 1.5625)
            scale_y = cfg_geom.get("scale_y_mm_per_px", 1.5625)
            self.homography = PlanarHomography(scale_x_mm_per_px=scale_x, scale_y_mm_per_px=scale_y)

        # Distance from camera origin to decision/ejection boundary in mm
        self.ejection_line_x_mm = CONFIG.get("pneumatics", {}).get("ejection_line_x_mm", 1200.0)

    def set_speed(self, new_speed_mps: float):
        """Allows dynamically updating belt speed without rebuilding pipeline."""
        self.speed_mps = max(0.1, min(10.0, new_speed_mps))

    def project_tracklet_metric(self, tracklet) -> Dict[str, Any]:
        """
        Maps a ByteTrack tracklet to real conveyor metric coordinates.
        Args:
            tracklet: STrack instance with to_tlwh() and polygon.
        Returns:
            dict containing metric position (x_w, y_w mm), metric area mm^2, velocity, and time to ejection.
        """
        tlwh = tracklet.to_tlwh()
        u_center = tlwh[0] + tlwh[2] / 2.0
        v_center = tlwh[1] + tlwh[3] / 2.0

        # Project centroid to metric space (mm)
        x_w, y_w = self.homography.pixel_to_world(u_center, v_center)

        # Calculate metric area from polygon or bounding box
        if hasattr(tracklet, "polygon") and tracklet.polygon and len(tracklet.polygon) >= 3:
            area_mm2 = self.homography.polygon_area_mm2(tracklet.polygon)
        else:
            # Bounding box metric width and height
            w_mm = tlwh[2] * self.homography.H[0, 0]
            h_mm = tlwh[3] * self.homography.H[1, 1]
            area_mm2 = abs(float(w_mm * h_mm))

        # Time remaining until item reaches ejection boundary
        # v_belt in mm/s = speed_mps * 1000.0
        v_mm_s = self.speed_mps * 1000.0
        dist_to_ejection_mm = max(0.0, self.ejection_line_x_mm - x_w)
        time_to_ejection_sec = dist_to_ejection_mm / v_mm_s if v_mm_s > 0 else 0.0

        return {
            "track_id": tracklet.track_id,
            "class_id": tracklet.class_id,
            "class_name": tracklet.class_name,
            "confidence": tracklet.score,
            "pixel_center": [float(u_center), float(v_center)],
            "metric_pos_mm": [round(x_w, 2), round(y_w, 2)],
            "metric_area_mm2": round(area_mm2, 2),
            "belt_speed_mps": self.speed_mps,
            "time_to_ejection_sec": round(time_to_ejection_sec, 4),
            "is_past_ejection": x_w >= self.ejection_line_x_mm
        }

    def simulate_conveyor_displacement(self, pixel_pos: Tuple[float, float], dt_seconds: float) -> Tuple[float, float]:
        """
        Calculates downstream pixel displacement after dt_seconds at current belt speed.
        Useful for continuous synthetic frame generation and tracking benchmarking.
        """
        u, v = pixel_pos
        # Linear motion along X axis (downstream): dx_mm = v_mps * 1000 * dt
        dx_mm = self.speed_mps * 1000.0 * dt_seconds
        dx_px = dx_mm / self.homography.H[0, 0]
        return (u + dx_px, v)
