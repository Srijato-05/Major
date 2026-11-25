import os
import json
import random
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from src.utils.config_loader import CONFIG


@dataclass
class BeltFragment:
    rgb_crop: np.ndarray
    spec_crop: np.ndarray
    mask_crop: np.ndarray
    x_px: float
    y_px: float
    w_px: int
    h_px: int


class ScrollingConveyorStream:
    """
    Continuous scrolling conveyor compositor using real held-out `test.json` e-waste items.
    Interleaves all foreground classes (1..6) and moves fragments smoothly downstream along +X
    at physical belt speed `speed_mps`, enabling genuine ByteTrack ID persistence and kinematics.
    """

    def __init__(
        self,
        manifest_path: str = "data/processed/test.json",
        frame_size: Tuple[int, int] = (640, 640),
        speed_mps: Optional[float] = None,
        fps: Optional[float] = None,
        scale_x_mm_per_px: Optional[float] = None,
        seed: int = 42,
    ):
        conv_cfg = CONFIG.get("conveyor", {})
        geom_cfg = CONFIG.get("optical_geometry", {})
        self.class_mapping = CONFIG.get("taxonomy", {}).get("dataset_mappings", {}).get("spectralwaste", {})

        self.width, self.height = frame_size
        self.speed_mps = float(speed_mps if speed_mps is not None else conv_cfg.get("speed_mps", 2.5))
        self.fps = float(fps if fps is not None else conv_cfg.get("sampling_rate_fps", 30))
        self.scale_x_mm_per_px = float(scale_x_mm_per_px if scale_x_mm_per_px is not None else geom_cfg.get("scale_x_mm_per_px", 1.5625))
        self.rng = random.Random(seed)

        self.records_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                for rec in json.load(f):
                    self.records_by_source[rec["source"]].append(rec)

        self._sources = sorted(self.records_by_source.keys())
        self._source_cursor = 0
        self._active_fragments: List[BeltFragment] = []
        self._belt_offset_px = 0.0
        self._spawn_cooldown_px = 0.0

    def set_speed(self, speed_mps: float):
        self.speed_mps = max(0.5, min(10.0, float(speed_mps)))

    def _extract_next_fragment(self) -> Optional[BeltFragment]:
        if not self._sources:
            return None

        for _ in range(len(self._sources) * 2):
            src = self._sources[self._source_cursor % len(self._sources)]
            self._source_cursor += 1
            recs = self.records_by_source[src]
            if not recs:
                continue
            rec = self.rng.choice(recs)

            rgb = cv2.imread(rec["rgb_path"], cv2.IMREAD_COLOR)
            spec = cv2.imread(rec["spectral_path"], cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(rec["mask_path"], cv2.IMREAD_UNCHANGED)
            if rgb is None or spec is None or mask is None:
                continue
            if spec.ndim == 3:
                spec = spec[:, :, 0]
            if mask.ndim == 3:
                mask = mask[:, :, 0]

            if src == "spectralwaste" and self.class_mapping:
                remapped = np.zeros_like(mask)
                for raw_id, target_id in self.class_mapping.items():
                    remapped[mask == int(raw_id)] = int(target_id)
                mask = remapped
            elif src == "battery_packs":
                mask = np.where(mask == 2, 3, mask)

            fg_ys, fg_xs = np.where(mask > 0)
            if fg_ys.size < 200:
                continue

            y1, y2 = int(fg_ys.min()), int(fg_ys.max()) + 1
            x1, x2 = int(fg_xs.min()), int(fg_xs.max()) + 1
            rgb_c = rgb[y1:y2, x1:x2]
            spec_c = spec[y1:y2, x1:x2]
            mask_c = mask[y1:y2, x1:x2]

            # Fit crop cleanly to conveyor fragment size [110..220 px]
            max_dim = max(rgb_c.shape[0], rgb_c.shape[1])
            target_dim = self.rng.randint(120, 210)
            scale = target_dim / max(max_dim, 1)
            new_w = max(48, int(round(rgb_c.shape[1] * scale)))
            new_h = max(48, int(round(rgb_c.shape[0] * scale)))

            rgb_c = cv2.resize(rgb_c, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            spec_c = cv2.resize(spec_c, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask_c = cv2.resize(mask_c, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            y_pos = float(self.rng.randint(24, max(25, self.height - new_h - 24)))
            return BeltFragment(
                rgb_crop=rgb_c,
                spec_crop=spec_c,
                mask_crop=mask_c,
                x_px=-float(new_w) + 16.0,
                y_px=y_pos,
                w_px=new_w,
                h_px=new_h,
            )
        return None

    def step(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advances belt by one frame (`dt = 1 / fps`) and composites RGB, spectral, and gt_mask."""
        dt = 1.0 / max(self.fps, 1.0)
        dx_mm = self.speed_mps * 1000.0 * dt
        dx_px = dx_mm / max(self.scale_x_mm_per_px, 0.1)

        self._belt_offset_px = (self._belt_offset_px + dx_px) % 48.0
        self._spawn_cooldown_px -= dx_px

        for frag in self._active_fragments:
            frag.x_px += dx_px
        self._active_fragments = [f for f in self._active_fragments if f.x_px < self.width]

        if self._spawn_cooldown_px <= 0.0 and len(self._active_fragments) < 4:
            new_frag = self._extract_next_fragment()
            if new_frag is not None:
                self._active_fragments.append(new_frag)
                self._spawn_cooldown_px = float(new_frag.w_px + self.rng.randint(45, 95))

        # Dark industrial belt canvas with subtle moving belt slat lines
        rgb_canvas = np.full((self.height, self.width, 3), 24, dtype=np.uint8)
        spec_canvas = np.full((self.height, self.width), 20, dtype=np.uint8)
        gt_mask = np.zeros((self.height, self.width), dtype=np.uint8)

        x_line = int(self._belt_offset_px)
        while x_line < self.width:
            cv2.line(rgb_canvas, (x_line, 0), (x_line, self.height), (34, 38, 44), 1)
            x_line += 48

        for frag in self._active_fragments:
            x0 = int(round(frag.x_px))
            y0 = int(round(frag.y_px))
            x1 = x0 + frag.w_px
            y1 = y0 + frag.h_px

            ix1, iy1 = max(0, x0), max(0, y0)
            ix2, iy2 = min(self.width, x1), min(self.height, y1)
            if ix2 <= ix1 or iy2 <= iy1:
                continue

            cx1, cy1 = ix1 - x0, iy1 - y0
            cx2, cy2 = cx1 + (ix2 - ix1), cy1 + (iy2 - iy1)

            fg_bool = frag.mask_crop[cy1:cy2, cx1:cx2] > 0
            rgb_roi = rgb_canvas[iy1:iy2, ix1:ix2]
            spec_roi = spec_canvas[iy1:iy2, ix1:ix2]
            mask_roi = gt_mask[iy1:iy2, ix1:ix2]

            rgb_roi[fg_bool] = frag.rgb_crop[cy1:cy2, cx1:cx2][fg_bool]
            spec_roi[fg_bool] = frag.spec_crop[cy1:cy2, cx1:cx2][fg_bool]
            mask_roi[fg_bool] = frag.mask_crop[cy1:cy2, cx1:cx2][fg_bool]

        # Convert BGR OpenCV canvas to RGB expected by pipeline
        rgb_out = cv2.cvtColor(rgb_canvas, cv2.COLOR_BGR2RGB)
        return rgb_out, spec_canvas, gt_mask

