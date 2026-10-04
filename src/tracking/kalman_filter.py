import numpy as np
from typing import Tuple

class ConveyorKalmanFilter:
    """
    Kalman filter for tracking bounding boxes in image space on a conveyor belt.
    State vector: [x, y, a, h, vx, vy, va, vh]
    - (x, y): Bounding box center coordinates
    - a: Aspect ratio (w / h)
    - h: Height
    - (vx, vy, va, vh): Corresponding velocities
    """

    def __init__(self):
        ndim, dt = 4, 1.0

        # State transition matrix
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Measurement projection matrix
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Standard deviations for uncertainty modeling
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create track from unassociated measurement.
        Args:
            measurement: [x, y, a, h] bounding box center, aspect ratio, height.
        Returns:
            mean: Initial state mean [8]
            covariance: Initial state covariance [8, 8]
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter prediction step.
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T
        )) + motion_cov

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project state distribution to measurement space.
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T
        )) + innovation_cov
        return mean, covariance

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter measurement correction step.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy_cho_factor(projected_cov)
        kalman_gain = scipy_cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T
        ))
        return new_mean, new_covariance


def scipy_cho_factor(matrix: np.ndarray):
    """Cholesky factorization with fallback to diagonal regularization."""
    try:
        from scipy.linalg import cho_factor
        return cho_factor(matrix, lower=True)
    except Exception:
        # Regularize
        reg_matrix = matrix + np.eye(matrix.shape[0]) * 1e-4
        from scipy.linalg import cho_factor
        return cho_factor(reg_matrix, lower=True)

def scipy_cho_solve(cho_tuple, b):
    from scipy.linalg import cho_solve
    return cho_solve(cho_tuple, b)
