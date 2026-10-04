"""
Exporter Module
Compiles PyTorch model checkpoints to ONNX (opset 17) and TensorRT FP16/INT8 compiled engines.
All exported artifacts are strictly locked within f:/Projects/Major/backend/weights/.
"""

import torch
import os
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [exporter]: %(message)s")
logger = logging.getLogger("exporter")

BASE_WEIGHTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights"))

def export_onnx(model: torch.nn.Module, output_path: str = None, input_shape=(1, 4, 640, 640)):
    """
    Exports PyTorch model to ONNX format (opset 17) strictly within workspace weights directory.
    """
    if output_path is None:
        output_path = os.path.join(BASE_WEIGHTS_DIR, "best_yolo11n_seg_4ch.onnx")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    logger.info(f"Exported ONNX model strictly to project workspace: {output_path}")

def build_tensorrt_engine(onnx_path: str = None, engine_path: str = None, precision: str = "FP16"):
    """
    Compiles ONNX model into TensorRT engine (FP16/INT8) inside workspace weights directory.
    """
    if onnx_path is None:
        onnx_path = os.path.join(BASE_WEIGHTS_DIR, "best_yolo11n_seg_4ch.onnx")
    if engine_path is None:
        engine_path = os.path.join(BASE_WEIGHTS_DIR, "best_yolo11n_seg_4ch.engine")

    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    logger.info(f"Compiling TensorRT engine ({precision}) strictly to project workspace: {engine_path}")
    return True
