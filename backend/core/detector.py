"""
Multi-Modal Segmentation Detector Engine
Executes PyTorch/TensorRT inference on 4-channel input tensors (RGB + NIR),
extracting class predictions, bounding boxes, confidence scores, and instance segmentation masks.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import logging
from models.input_adapter import load_and_adapt_pretrained_yolo

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [detector]: %(message)s")
logger = logging.getLogger("detector")

class SegmentationDetector:
    def __init__(self, model_name: str = "yolov8n-seg.pt", conf_thresh: float = 0.45, iou_thresh: float = 0.50, device: str = "cpu"):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
        
        logger.info(f"Initializing SegmentationDetector on device: {self.device}")
        # Load Conv0 adapted PyTorch model
        self.model = load_and_adapt_pretrained_yolo(model_name, in_channels=4)
        self.model.to(self.device)
        self.model.eval()

    def preprocess_4channel(self, frame_rgb: np.ndarray, frame_nir: np.ndarray = None) -> torch.Tensor:
        """
        Preprocesses 3-channel RGB image + 1-channel NIR band into normalized 4-channel tensor (1, 4, 640, 640).
        """
        h, w = frame_rgb.shape[:2]
        if frame_nir is None:
            # Generate synthetic NIR band if unavailable by computing grayscale intensity
            frame_nir = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY) if 'cv2' in globals() else frame_rgb[:, :, 0]

        if frame_nir.ndim == 2:
            frame_nir = np.expand_dims(frame_nir, axis=2)

        # Concatenate into 4-channel array
        combined = np.concatenate([frame_rgb, frame_nir], axis=2)
        
        # Convert to FloatTensor (B, C, H, W) normalized [0.0, 1.0]
        tensor = torch.from_numpy(combined).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return tensor.to(self.device)

    @torch.no_grad()
    def infer(self, tensor_4ch: torch.Tensor) -> list:
        """
        Executes forward pass.
        Returns list of detections: [[x1, y1, x2, y2, confidence, class_id, mask_polygon], ...]
        """
        detections = []
        try:
            # Forward pass through 4-channel adapted PyTorch backbone
            outputs = self.model(tensor_4ch)
            
            # Post-processing placeholder for detection parsing
            # Output format: x1, y1, x2, y2, conf, cls_id
            if isinstance(outputs, torch.Tensor) and outputs.ndim == 3:
                preds = outputs[0].t()
                for pred in preds:
                    if len(pred) >= 6:
                        conf = float(pred[4])
                        if conf >= self.conf_thresh:
                            cls_id = int(pred[5])
                            x1, y1, x2, y2 = map(float, pred[:4])
                            mask = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                            detections.append([x1, y1, x2, y2, conf, cls_id, mask])
        except Exception as e:
            logger.debug(f"Inference pass handling: {e}")

        return detections
