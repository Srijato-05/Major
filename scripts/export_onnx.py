import os
import sys
import time
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from src.utils.config_loader import CONFIG
from src.models.multimodal_segmenter import build_4channel_segmentation_model


class OnnxExportWrapper(nn.Module):
    def __init__(self, seg_model: nn.Module):
        super().__init__()
        self.seg_model = seg_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seg_model(x)["out"]


def benchmark_latency(model: nn.Module, device: torch.device, img_size: int, use_fp16: bool, iters: int = 30):
    x = torch.randn(1, 4, img_size, img_size, device=device)
    with torch.no_grad():
        for _ in range(10):
            with torch.amp.autocast("cuda", enabled=(use_fp16 and device.type == "cuda")):
                _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.amp.autocast("cuda", enabled=(use_fp16 and device.type == "cuda")):
                _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
    ms = dt * 1000.0
    fps = 1000.0 / ms if ms > 0 else 0.0
    return ms, fps


def main():
    default_size = int(CONFIG.get("model", {}).get("img_size", 384))
    parser = argparse.ArgumentParser(description="Export 4-channel DeepLabV3 model to ONNX and benchmark FP32 vs FP16.")
    parser.add_argument("--weights", default="models/checkpoints/multimodal_4ch_best.pth")
    parser.add_argument("--onnx-out", default="models/checkpoints/multimodal_4ch_best.onnx")
    parser.add_argument("--img-size", type=int, default=default_size)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_4channel_segmentation_model(num_classes=7, in_channels=4, pretrained=False)
    if os.path.exists(args.weights):
        chk = torch.load(args.weights, map_location="cpu", weights_only=False)
        sd = chk.get("model_state_dict", chk)
        model_sd = model.state_dict()
        filtered = {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}
        model.load_state_dict(filtered, strict=False)
        print(f"Loaded checkpoint ({len(filtered)}/{len(model_sd)} tensors): {args.weights}")

    model.eval()
    wrapper = OnnxExportWrapper(model).eval()
    dummy = torch.randn(1, 4, args.img_size, args.img_size)
    os.makedirs(os.path.dirname(args.onnx_out), exist_ok=True)
    try:
        torch.onnx.export(
            wrapper,
            dummy,
            args.onnx_out,
            opset_version=17,
            input_names=["input_4ch"],
            output_names=["logits"],
            dynamic_axes={"input_4ch": {0: "batch_size"}, "logits": {0: "batch_size"}},
        )
        size_mb = os.path.getsize(args.onnx_out) / (1024 ** 2)
        print(f"[ONNX EXPORT] Saved {args.onnx_out} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"[ONNX SKIPPED] {e} (run `pip install onnx` to enable .onnx export)")
        ts_path = os.path.splitext(args.onnx_out)[0] + ".pt"
        traced = torch.jit.trace(wrapper, dummy)
        traced.save(ts_path)
        size_mb = os.path.getsize(ts_path) / (1024 ** 2)
        print(f"[TORCHSCRIPT EXPORT] Saved {ts_path} ({size_mb:.2f} MB)")

    model.to(device)
    ms_fp32, fps_fp32 = benchmark_latency(model, device, args.img_size, use_fp16=False)
    print(f"[BENCHMARK FP32] {ms_fp32:.2f} ms/frame ({fps_fp32:.1f} FPS) on {device}")
    if device.type == "cuda":
        ms_fp16, fps_fp16 = benchmark_latency(model, device, args.img_size, use_fp16=True)
        print(f"[BENCHMARK FP16] {ms_fp16:.2f} ms/frame ({fps_fp16:.1f} FPS) on {device}")


if __name__ == "__main__":
    main()
