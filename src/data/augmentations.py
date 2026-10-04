import numpy as np
import cv2
import random
from typing import Dict, Any, Optional

class IndustrialMultimodalAugmentation:
    """
    Applies realistic industrial conveyor degradations to 4-channel tensors:
    1. Directional Motion Blur: Aligned with conveyor belt travel axis (1.0 to 4.0 m/s)
    2. Specular Glare Injection: Simulates harsh overhead lighting reflections on metals
    3. Industrial Dust Masking: Perlin/Gaussian particulate contamination
    4. CutOut / Random Erasing: Models partial occlusion on dense conveyor lines
    """

    def __init__(
        self,
        belt_speed_mps: Optional[float] = None,
        blur_prob: Optional[float] = None,
        glare_prob: Optional[float] = None,
        dust_prob: Optional[float] = None,
        cutout_prob: Optional[float] = None
    ):
        from src.utils.config_loader import CONFIG
        aug_cfg = CONFIG.get("training", {}).get("augmentations", {})
        conveyor_cfg = CONFIG.get("conveyor", {})

        self.belt_speed_mps = belt_speed_mps if belt_speed_mps is not None else conveyor_cfg.get("default_speed_mps", 2.5)
        self.blur_prob = blur_prob if blur_prob is not None else aug_cfg.get("blur_prob", 0.5)
        self.glare_prob = glare_prob if glare_prob is not None else aug_cfg.get("glare_prob", 0.4)
        self.dust_prob = dust_prob if dust_prob is not None else aug_cfg.get("dust_prob", 0.3)
        self.cutout_prob = cutout_prob if cutout_prob is not None else aug_cfg.get("cutout_prob", 0.3)

    def apply_motion_blur(self, image_4ch: np.ndarray) -> np.ndarray:
        """
        Applies directional linear convolution kernel:
        k in [15, 31] pixels aligned along movement axis with small angular jitter theta in [-2 deg, +2 deg].
        """
        # Kernel size scales with conveyor speed (e.g. 15 px at 1.0 m/s up to 31 px at 4.0 m/s)
        k_size = int(15 + (self.belt_speed_mps - 1.0) / 3.0 * 16)
        k_size = max(3, k_size if k_size % 2 == 1 else k_size + 1)

        # Build horizontal motion blur kernel
        kernel = np.zeros((k_size, k_size), dtype=np.float32)
        angle = random.uniform(-2.0, 2.0)
        center = k_size // 2

        # Draw a line through the center
        for x in range(k_size):
            y = int(center + (x - center) * np.tan(np.radians(angle)))
            if 0 <= y < k_size:
                kernel[y, x] = 1.0

        k_sum = kernel.sum()
        if k_sum > 0:
            kernel /= k_sum
        else:
            kernel[center, center] = 1.0

        return cv2.filter2D(image_4ch, -1, kernel)

    def apply_specular_glare(self, image_4ch: np.ndarray) -> np.ndarray:
        """
        Superimposes high-intensity elliptical Gaussian spots saturating RGB and NIR channels:
        I_glare = A * exp(-((x - x0)^2 / (2*sigma_x^2) + (y - y0)^2 / (2*sigma_y^2)))
        """
        h, w = image_4ch.shape[:2]
        x0 = random.randint(0, w - 1)
        y0 = random.randint(0, h - 1)
        sigma_x = random.uniform(15.0, 45.0)
        sigma_y = random.uniform(15.0, 45.0)
        amplitude = random.uniform(180.0, 255.0)

        y, x = np.ogrid[:h, :w]
        glare_spot = amplitude * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2)))

        out = image_4ch.astype(np.float32).copy()
        # Glare primarily affects RGB and NIR reflection
        for c in range(min(4, out.shape[2])):
            out[:, :, c] = np.clip(out[:, :, c] + glare_spot, 0, 255)

        return out.astype(np.uint8)

    def apply_industrial_dust(self, image_4ch: np.ndarray) -> np.ndarray:
        """
        Simulates particulate deposition on optical lenses using localized noise modulation.
        """
        h, w, c = image_4ch.shape
        noise = np.random.normal(0, 15, (h, w, 1))
        out = image_4ch.astype(np.float32) + noise
        return np.clip(out, 0, 255).astype(np.uint8)

    def apply_cutout(self, image_4ch: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Applies random rectangular erasure up to 30% area to simulate scrap item overlap.
        """
        h, w = image_4ch.shape[:2]
        cut_h = int(h * random.uniform(0.1, 0.3))
        cut_w = int(w * random.uniform(0.1, 0.3))
        y0 = random.randint(0, h - cut_h)
        x0 = random.randint(0, w - cut_w)

        out = image_4ch.copy()
        out[y0:y0 + cut_h, x0:x0 + cut_w, :] = 0
        if mask is not None:
            mask[y0:y0 + cut_h, x0:x0 + cut_w] = 0
        return out

    def __call__(self, image_4ch: np.ndarray, mask: np.ndarray = None) -> Dict[str, np.ndarray]:
        out = image_4ch.copy()
        out_mask = mask.copy() if mask is not None else None

        if random.random() < self.blur_prob:
            out = self.apply_motion_blur(out)
        if random.random() < self.glare_prob:
            out = self.apply_specular_glare(out)
        if random.random() < self.dust_prob:
            out = self.apply_industrial_dust(out)
        if random.random() < self.cutout_prob:
            out = self.apply_cutout(out, out_mask)

        return {"image": out, "mask": out_mask}
