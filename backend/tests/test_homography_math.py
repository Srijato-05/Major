"""
Unit Test: 2D Planar Homography Coordinate Transformation Precision
"""

import pytest
import numpy as np
from core.kinematics import HomographyKinematicsMapper

def test_homography_transformation_precision():
    src_pts = np.array([[42, 38], [598, 40], [630, 610], [10, 608]], dtype=np.float64)
    dst_pts = np.array([[0.0, 0.0], [0.8, 0.0], [0.8, 1.2], [0.0, 1.2]], dtype=np.float64)

    mapper = HomographyKinematicsMapper(src_pts, dst_pts)

    for i in range(4):
        u, v = src_pts[i]
        expected_x, expected_y = dst_pts[i]
        calc_x, calc_y = mapper.pixel_to_world(u, v)

        assert abs(calc_x - expected_x) < 1e-3, f"X Error: {calc_x} vs {expected_x}"
        assert abs(calc_y - expected_y) < 1e-3, f"Y Error: {calc_y} vs {expected_y}"
