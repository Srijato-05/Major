"""
Industrial Augmenter
Albumentations pipeline for simulating conveyor motion blur, overhead LED glare, lens dust, and random occlusions.
"""

import cv2
import numpy as np

def apply_motion_blur(image: np.ndarray, speed_m_s: float = 2.5) -> np.ndarray:
    """
    Applies linear motion blur aligned with conveyor movement vector.
    """
    kernel_size = int(max(3, min(31, speed_m_s * 6)))
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = 1.0 / kernel_size
    return cv2.filter2D(image, -1, kernel)

def inject_specular_glare(image: np.ndarray) -> np.ndarray:
    """
    Injects high-intensity Gaussian specular glare spots onto target image channels.
    """
    h, w = image.shape[:2]
    cx, cy = np.random.randint(0, w), np.random.randint(0, h)
    glare_mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(glare_mask, (cx, cy), np.random.randint(10, 40), 255, -1)
    glare_mask = cv2.GaussianBlur(glare_mask, (51, 51), 0)
    
    output = image.astype(np.float32)
    for c in range(image.shape[2] if image.ndim == 3 else 1):
        if image.ndim == 3:
            output[:, :, c] = np.clip(output[:, :, c] + glare_mask, 0, 255)
        else:
            output = np.clip(output + glare_mask, 0, 255)
    return output.astype(np.uint8)
