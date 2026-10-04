import numpy as np
import cv2
from typing import Tuple, List, Optional

class PlanarHomography:
    """
    2D Planar Homography Transformation Matrix (H in R^(3x3))
    Transforms pixel coordinates (u, v) on the camera image plane into
    real-world metric coordinates (x_w, y_w) in millimeters on the conveyor belt.
    """

    def __init__(
        self,
        scale_x_mm_per_px: float = 1.5625,
        scale_y_mm_per_px: float = 1.5625,
        origin_offset_x_mm: float = 0.0,
        origin_offset_y_mm: float = 0.0,
        matrix_h: Optional[np.ndarray] = None
    ):
        if matrix_h is not None:
            self.H = np.array(matrix_h, dtype=np.float64)
        else:
            # Construct standard affine projection matrix
            self.H = np.array([
                [scale_x_mm_per_px, 0.0, origin_offset_x_mm],
                [0.0, scale_y_mm_per_px, origin_offset_y_mm],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)

        self.H_inv = np.linalg.inv(self.H)

    @classmethod
    def from_calibration_points(
        cls,
        image_points: List[Tuple[float, float]],
        world_points: List[Tuple[float, float]]
    ) -> "PlanarHomography":
        """
        Solves the transformation matrix H using Direct Linear Transform (DLT)
        from N >= 4 non-collinear reference points (e.g. ArUco markers on conveyor).
        """
        pts_img = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
        pts_world = np.array(world_points, dtype=np.float32).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(pts_img, pts_world, method=0)
        return cls(matrix_h=H)

    def pixel_to_world(self, u: float, v: float) -> Tuple[float, float]:
        """
        Projects pixel coordinates (u, v) to conveyor metric coordinates (x_w, y_w) in mm.
        """
        px_homog = np.array([u, v, 1.0], dtype=np.float64)
        world_homog = self.H @ px_homog
        x_w = world_homog[0] / world_homog[2]
        y_w = world_homog[1] / world_homog[2]
        return float(x_w), float(y_w)

    def world_to_pixel(self, x_w: float, y_w: float) -> Tuple[float, float]:
        """
        Projects conveyor metric coordinates (x_w, y_w) in mm back to image pixels (u, v).
        """
        world_homog = np.array([x_w, y_w, 1.0], dtype=np.float64)
        px_homog = self.H_inv @ world_homog
        u = px_homog[0] / px_homog[2]
        v = px_homog[1] / px_homog[2]
        return float(u), float(v)

    def polygon_area_mm2(self, pixel_polygon: List[Tuple[float, float]]) -> float:
        """
        Calculates the real surface area (in mm^2) of a segmented polygon mask
        after projecting all contour vertices into metric conveyor space.
        """
        if len(pixel_polygon) < 3:
            return 0.0

        world_pts = [self.pixel_to_world(pt[0], pt[1]) for pt in pixel_polygon]
        world_pts_np = np.array(world_pts, dtype=np.float32)

        area = cv2.contourArea(world_pts_np)
        return float(area)
