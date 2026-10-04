"""
Planar Homography & Metric Kinematics Mapper
Corrects radial and tangential camera lens distortion and maps 2D image coordinates (u, v)
to 2D planar conveyor metric world coordinates (x_w, y_w) using SVD Homography matrix H.
"""

import numpy as np

class HomographyKinematicsMapper:
    def __init__(self, src_pts: np.ndarray, dst_pts: np.ndarray, dist_coeffs: list = None):
        """
        src_pts: 4 camera pixel coordinates [[u0, v0], [u1, v1], [u2, v2], [u3, v3]]
        dst_pts: 4 metric world coordinates [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
        dist_coeffs: Optional lens distortion coefficients [k1, k2, p1, p2, k3]
        """
        self.src_pts = np.array(src_pts, dtype=np.float64)
        self.dst_pts = np.array(dst_pts, dtype=np.float64)
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else [0.0, 0.0, 0.0, 0.0, 0.0]
        self.H = self._compute_homography_svd(self.src_pts, self.dst_pts)

    def _compute_homography_svd(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """
        Solves Direct Linear Transform system A * h = 0 via Singular Value Decomposition (SVD).
        """
        A = []
        for i in range(4):
            u, v = src[i]
            x, y = dst[i]
            A.append([-u, -v, -1, 0, 0, 0, u * x, v * x, x])
            A.append([0, 0, 0, -u, -v, -1, u * y, v * y, y])
        A = np.array(A, dtype=np.float64)
        
        # Singular Value Decomposition
        U, S, Vh = np.linalg.svd(A)
        # Optimal homography corresponds to last row of Vh (smallest singular value)
        H = Vh[-1].reshape((3, 3))
        return H / H[2, 2]

    def correct_lens_distortion(self, u: float, v: float, cx: float = 320.0, cy: float = 320.0, fx: float = 600.0, fy: float = 600.0) -> tuple:
        """
        Applies radial (k1, k2, k3) and tangential (p1, p2) lens distortion correction math.
        """
        k1, k2, p1, p2, k3 = self.dist_coeffs
        if k1 == 0.0 and k2 == 0.0 and p1 == 0.0 and p2 == 0.0:
            return u, v

        x = (u - cx) / fx
        y = (v - cy) / fy
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2

        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        x_corr = x * radial + (2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x))
        y_corr = y * radial + (p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y)

        u_corr = x_corr * fx + cx
        v_corr = y_corr * fy + cy
        return u_corr, v_corr

    def pixel_to_world(self, u: float, v: float) -> tuple:
        """
        Maps pixel coordinate (u, v) to conveyor metric world coordinate (x_w, y_w).
        """
        u_corr, v_corr = self.correct_lens_distortion(u, v)
        pt = np.array([u_corr, v_corr, 1.0], dtype=np.float64)
        dst = self.H @ pt
        x_w = dst[0] / dst[2]
        y_w = dst[1] / dst[2]
        return float(x_w), float(y_w)
