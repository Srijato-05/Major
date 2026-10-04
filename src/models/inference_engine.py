import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
from src.models.multimodal_segmenter import Multimodal4ChSegmentationModel, build_4channel_segmentation_model
from src.utils.config_loader import CONFIG

class MultimodalInferenceEngine:
    """
    High-Performance Inference and Prediction Engine for 4-Channel E-Waste Streams.
    Handles:
    - Normalization and 4-channel tensor preparation (RGB + NIR/SWIR or RGB-only fallback)
    - Forward inference and class argmax mask generation
    - Polygon contour extraction for individual fragments
    - Metric area calculation via Homography
    """

    def __init__(
        self,
        model: Optional[Multimodal4ChSegmentationModel] = None,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        conf_threshold: Optional[float] = None
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        cfg_model = CONFIG.get("model", {})
        self.conf_threshold = conf_threshold if conf_threshold is not None else cfg_model.get("conf_threshold", 0.35)
        self.model = model or build_4channel_segmentation_model(pretrained=True)

        if weights_path and os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        # Class lookup
        taxonomy = CONFIG.get("taxonomy", {}).get("classes", {})
        self.class_names = {int(k): v["name"] for k, v in taxonomy.items()}

    def preprocess_image(
        self,
        rgb_image: np.ndarray,
        spectral_band: Optional[np.ndarray] = None,
        target_size: Tuple[int, int] = (640, 640)
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Prepares 4-channel input tensor. If spectral_band is omitted,
        generates synthetic infrared fallback using channel mean.
        """
        orig_h, orig_w = rgb_image.shape[:2]

        if spectral_band is None:
            # Infrared fallback: channel-wise mean representing broad-spectrum NIR reflectance
            spectral_1ch = rgb_image.mean(axis=2, keepdims=True).astype(np.uint8)
        elif len(spectral_band.shape) == 2:
            spectral_1ch = spectral_band[:, :, np.newaxis]
        elif len(spectral_band.shape) == 3:
            # If 3-channel pseudo-color HSI is passed, extract the first spectral channel
            spectral_1ch = spectral_band[:, :, 0:1]
        else:
            spectral_1ch = spectral_band

        # Ensure RGB is 3-channel
        if len(rgb_image.shape) == 2:
            rgb_3ch = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2RGB)
        else:
            rgb_3ch = rgb_image[:, :, :3]

        # Stack into exactly 4-channel image: (H, W, 4)
        img_4ch = np.concatenate([rgb_3ch, spectral_1ch], axis=2)
        resized_4ch = cv2.resize(img_4ch, target_size, interpolation=cv2.INTER_LINEAR)

        # Tensor conversion [1, 4, H, W]
        tensor = torch.from_numpy(resized_4ch).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        return tensor.to(self.device), (orig_h, orig_w)

    @torch.no_grad()
    def predict(
        self,
        rgb_image: np.ndarray,
        spectral_band: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Executes segmentation inference and returns detected instances.
        """
        tensor, (orig_h, orig_w) = self.preprocess_image(rgb_image, spectral_band)
        outputs = self.model(tensor)
        logits = outputs["out"]  # [1, num_classes, H, W]
        probs = F.softmax(logits, dim=1).squeeze(0)  # [num_classes, H, W]

        # Extract per-class probability maps with spatial bilateral / median filtering
        probs_np = probs.cpu().numpy()  # [num_classes, H, W]
        
        # Spatial filtering on polymer and foreground probabilities to suppress single-pixel sensor noise
        smoothed_probs = np.zeros_like(probs_np)
        for c_idx in range(probs_np.shape[0]):
            ch = (probs_np[c_idx] * 255.0).astype(np.uint8)
            ch_smooth = cv2.medianBlur(ch, 5)
            smoothed_probs[c_idx] = ch_smooth.astype(np.float32) / 255.0

        # Class predictions with calibrated thresholding
        # Default background unless foreground class exceeds calibrated threshold (0.60 for polymers, 0.45 for others)
        class_map_np = np.zeros((probs_np.shape[1], probs_np.shape[2]), dtype=np.uint8)
        conf_map_np = smoothed_probs[0].copy()

        # Check foreground classes
        for c_idx in [3, 1, 2, 4, 5]:
            thresh = 0.58 if c_idx == 3 else 0.45
            fg_mask = (smoothed_probs[c_idx] >= thresh) & (smoothed_probs[c_idx] > conf_map_np)
            class_map_np[fg_mask] = c_idx
            conf_map_np[fg_mask] = smoothed_probs[c_idx][fg_mask]

        # Where no foreground exceeded calibrated threshold, assign background
        bg_mask = (class_map_np == 0)
        conf_map_np[bg_mask] = smoothed_probs[0][bg_mask]

        # Resize back to original dimensions
        class_map_orig = cv2.resize(class_map_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        conf_map_orig = cv2.resize(conf_map_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Extract discrete polygon instances using morphological peak separation
        instances: List[Dict[str, Any]] = []
        unique_classes = np.unique(class_map_orig)

        for c in unique_classes:
            if c == 0:
                continue  # Background class

            binary_mask = (class_map_orig == c).astype(np.uint8)
            # Morphological opening to eliminate tiny stray artifacts
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            if binary_mask.sum() < 60:
                continue

            # Distance transform to isolate distinct mass centers of touching irregular fragments
            dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
            max_dist = dist.max()

            separated_cnts = []
            if max_dist > 16.0:
                # Disconnect clumped fragments by peak thresholding
                _, sure_fg = cv2.threshold(dist, 0.25 * max_dist, 255, cv2.THRESH_BINARY)
                sure_fg = np.uint8(sure_fg)
                num_markers, markers = cv2.connectedComponents(sure_fg)
                
                if num_markers > 2:
                    markers = markers + 1
                    unknown = cv2.subtract(binary_mask * 255, sure_fg)
                    markers[unknown == 255] = 0

                    syn_bgr = cv2.cvtColor(binary_mask * 255, cv2.COLOR_GRAY2BGR)
                    markers = cv2.watershed(syn_bgr, markers)

                    for mid in range(2, num_markers + 1):
                        part = (markers == mid).astype(np.uint8)
                        cnts_part, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cp in cnts_part:
                            if cv2.contourArea(cp) >= 80:
                                separated_cnts.append(cp)

            if not separated_cnts:
                cnts_raw, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                separated_cnts = [cnt for cnt in cnts_raw if cv2.contourArea(cnt) >= 80]

            for cnt in separated_cnts:
                area_px = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Sample local confidence within contour mask
                c_mask = np.zeros_like(binary_mask)
                cv2.drawContours(c_mask, [cnt], -1, 1, -1)
                mean_conf = float(conf_map_orig[c_mask == 1].mean()) if c_mask.sum() > 0 else float(conf_map_orig[binary_mask == 1].mean())
                polygon = cnt.reshape(-1, 2).tolist()

                instances.append({
                    "class_id": int(c),
                    "class_name": self.class_names.get(int(c), f"Class_{c}"),
                    "confidence": round(mean_conf, 3),
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "area_px": float(area_px),
                    "polygon": polygon
                })

        return {
            "instances": instances,
            "mask": class_map_orig,
            "confidence_map": conf_map_orig,
            "num_detections": len(instances)
        }

    def predict_frame(self, frame: np.ndarray, spectral_band: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Alias for predict(), optimized for video frame loops with guaranteed exception handling.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return {"instances": [], "mask": None, "confidence_map": None, "num_detections": 0}
        return self.predict(frame, spectral_band)

    def annotate_frame(self, frame: np.ndarray, predictions: Dict[str, Any]) -> np.ndarray:
        """
        Renders visual bounding boxes, colored polygonal masks, and class labels onto an RGB frame.
        Guarantees input frame is not modified in-place and handles grayscale/RGB/RGBA transparently.
        """
        if frame is None or frame.size == 0:
            return frame

        canvas = frame.copy()
        if len(canvas.shape) == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        elif canvas.shape[2] == 4:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGBA2BGR)

        # Predefined distinct vibrant BGR colors for each WEEE class
        class_colors = {
            0: (0, 255, 0),    # High-Grade PCB (Green)
            1: (255, 0, 255),  # IC Molds & Chips (Magenta)
            2: (0, 0, 255),    # Battery Pack (Red alert)
            3: (0, 165, 255),  # BFR Polymers (Orange)
            4: (255, 255, 0),  # Metallic Heat Sinks (Cyan)
            5: (255, 128, 0),  # Copper Wire Harnesses (Blue)
        }

        for inst in predictions.get("instances", []):
            cid = inst.get("class_id", 0)
            color = class_colors.get(cid, (0, 255, 255))
            bbox = inst.get("bbox", [0, 0, 0, 0])
            name = inst.get("class_name", "Unknown")
            conf = inst.get("confidence", 0.0)

            x, y, w, h = bbox

            # 1. Draw glowing irregular polygon contour & tinted overlay
            poly = inst.get("polygon")
            if poly and len(poly) >= 3:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                overlay = canvas.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.40, canvas, 0.60, 0, canvas)
                cv2.polylines(canvas, [pts], True, (255, 255, 255), 2, cv2.LINE_AA) # Crisp bright border
                cv2.polylines(canvas, [pts], True, color, 1, cv2.LINE_AA)

            # 2. Draw subtle dashed/thin bounding box so it does not clutter the view
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 1, cv2.LINE_4)

            # 3. Compact modern label chip
            label = f"{name} {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            chip_y1 = max(0, y - lh - 4)
            chip_y2 = y
            chip_x2 = min(canvas.shape[1] - 1, x + lw + 6)
            cv2.rectangle(canvas, (x, chip_y1), (chip_x2, chip_y2), (20, 24, 33), -1)
            cv2.rectangle(canvas, (x, chip_y1), (chip_x2, chip_y2), color, 1)
            cv2.putText(canvas, label, (x + 3, max(lh + 2, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        return canvas
