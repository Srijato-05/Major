import os
import re
import argparse
import json
import urllib.request
import zipfile
import shutil
import glob
import cv2
import numpy as np
from typing import Dict, Any, Optional

ZENODO_API_URL = "https://zenodo.org/api/records/10617721"

ENVI_DTYPE_MAP = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    12: np.uint16,
    13: np.uint32,
}


def download_with_progress(url: str, dest_path: str) -> None:
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024:
        print(f"Skipping existing file: {dest_path}")
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_path = dest_path + ".part"
    print(f"Downloading {os.path.basename(dest_path)} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_path, "wb") as out:
        shutil.copyfileobj(resp, out, length=1024 * 1024)
    os.replace(tmp_path, dest_path)


def download_pcb_vision(raw_dir: str = "data/raw/pcb_vision") -> None:
    os.makedirs(raw_dir, exist_ok=True)
    req = urllib.request.Request(ZENODO_API_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        record = json.loads(resp.read().decode("utf-8"))

    files = record.get("files", [])
    print(f"[PCB-VISION] Found {len(files)} archive file(s) on Zenodo record 10617721.")
    for fmeta in files:
        fname = fmeta["key"]
        url = fmeta["links"]["self"]
        dest = os.path.join(raw_dir, fname)
        download_with_progress(url, dest)
        if fname.lower().endswith(".zip"):
            extract_dir = os.path.join(raw_dir, "extracted")
            marker = os.path.join(extract_dir, f".done_{fname}")
            if not os.path.exists(marker):
                print(f"Extracting {fname}...")
                os.makedirs(extract_dir, exist_ok=True)
                with zipfile.ZipFile(dest, "r") as zf:
                    zf.extractall(extract_dir)
                open(marker, "w").close()


def parse_envi_hdr(hdr_path: str) -> Dict[str, Any]:
    """Parses an ENVI .hdr / .HDR metadata file."""
    with open(hdr_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Collapse multi-line brace blocks
    text = re.sub(r"\{([^}]*)\}", lambda m: "{" + m.group(1).replace("\n", " ").replace("\r", " ") + "}", text, flags=re.DOTALL)
    meta: Dict[str, Any] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        meta[k.strip().lower()] = v.strip()
    return meta


def read_envi_cube(hdr_path: str) -> Optional[np.ndarray]:
    """Reads an ENVI binary cube (bsq/bil/bip) returning shape (lines, samples, bands)."""
    meta = parse_envi_hdr(hdr_path)
    lines = int(meta["lines"])
    samples = int(meta["samples"])
    bands = int(meta["bands"])
    dtype_code = int(meta.get("data type", 4))
    np_dtype = np.dtype(ENVI_DTYPE_MAP[dtype_code])
    byte_order = int(meta.get("byte order", 0))
    np_dtype = np_dtype.newbyteorder(">" if byte_order == 1 else "<")
    offset = int(meta.get("header offset", 0))
    interleave = meta.get("interleave", "bsq").lower()

    stem = os.path.splitext(hdr_path)[0]
    if not os.path.exists(stem):
        return None

    with open(stem, "rb") as f:
        f.seek(offset)
        arr = np.fromfile(f, dtype=np_dtype, count=lines * samples * bands)

    if arr.size != lines * samples * bands:
        return None

    if interleave == "bsq":
        cube = arr.reshape((bands, lines, samples)).transpose(1, 2, 0)
    elif interleave == "bil":
        cube = arr.reshape((lines, bands, samples)).transpose(0, 2, 1)
    elif interleave == "bip":
        cube = arr.reshape((lines, samples, bands))
    else:
        return None
    return cube


def _percentile_u8(arr: np.ndarray) -> np.ndarray:
    arr_f = arr.astype(np.float32)
    lo, hi = np.percentile(arr_f, [2.0, 98.0])
    if hi <= lo:
        return np.zeros(arr_f.shape, dtype=np.uint8)
    scaled = np.clip((arr_f - lo) / (hi - lo) * 255.0, 0, 255)
    return scaled.astype(np.uint8)


def _find_dataset_root(raw_dir: str) -> Optional[str]:
    for root, dirs, _ in os.walk(raw_dir):
        if "__MACOSX" in root:
            continue
        if "HSI" in dirs and "RGB" in dirs:
            return root
    return None


def tile_pcb_vision(raw_dir: str = "data/raw/pcb_vision", tile_size: int = 512, stride: int = 512) -> int:
    """
    Processes both PCB-Vision modalities into harmonized 7-class triplets:
    1) ENVI HSI cubes (true 850-950 nm NIR + synthesized RGB + HSI/General_masks)
    2) High-res DSLR RGB scenes + RGB/General masks (tiled into 512x512 patches)
    Class mapping: 1 (IC) -> Class 2, 2/3 (Capacitor/Connectors) & PCB substrate -> Class 1.
    """
    tiles_dir = os.path.join(raw_dir, "tiles")
    rgb_out = os.path.join(tiles_dir, "rgb")
    nir_out = os.path.join(tiles_dir, "nir")
    mask_out = os.path.join(tiles_dir, "masks")
    os.makedirs(rgb_out, exist_ok=True)
    os.makedirs(nir_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    ds_root = _find_dataset_root(raw_dir)
    if ds_root is None:
        print(f"[PCB-VISION] Dataset root (HSI/RGB) not found under {raw_dir}.")
        return 0

    count = 0

    # --- A. Process all 53 ENVI HSI cubes + pixel-aligned ENVI General_masks ---
    hsi_dir = os.path.join(ds_root, "HSI")
    gen_dir = os.path.join(hsi_dir, "General_masks")
    for idx in range(1, 100):
        hdr_path = os.path.join(hsi_dir, f"pcb{idx}", f"pcb{idx}.hdr")
        mask_hdr = os.path.join(gen_dir, f"{idx}.HDR")
        if not os.path.exists(hdr_path):
            continue

        cube = read_envi_cube(hdr_path)
        if cube is None or cube.shape[2] < 200:
            continue

        # Default RGB bands (1-indexed {89, 64, 28} -> 0-indexed [88, 63, 27]) -> BGR for cv2.imwrite
        r_ch = _percentile_u8(cube[:, :, 88])
        g_ch = _percentile_u8(cube[:, :, 63])
        b_ch = _percentile_u8(cube[:, :, 27])
        bgr_img = np.stack([b_ch, g_ch, r_ch], axis=2)

        # True VNIR ~850-950 nm NIR band (Specim FX10 bands 165:210)
        nir_img = _percentile_u8(cube[:, :, 165:210].mean(axis=2))

        # Build 7-class mask: PCB substrate from non-dark foreground + IC/components from General_masks
        gray_val = bgr_img.mean(axis=2)
        mask = np.where(gray_val > 25, 1, 0).astype(np.uint8)
        if os.path.exists(mask_hdr):
            envi_mask = read_envi_cube(mask_hdr)
            if envi_mask is not None and envi_mask.shape[:2] == mask.shape:
                em = envi_mask[:, :, 0]
                mask[em == 1] = 2  # IC -> Class 2 (IC Molds & Chips)
                mask[(em == 2) | (em == 3)] = 1  # Capacitors / Connectors -> Class 1 (PCB)

        bgr_rs = cv2.resize(bgr_img, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
        nir_rs = cv2.resize(nir_img, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
        mask_rs = cv2.resize(mask, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)

        out_name = f"hsi_pcb{idx:03d}.png"
        cv2.imwrite(os.path.join(rgb_out, out_name), bgr_rs)
        cv2.imwrite(os.path.join(nir_out, out_name), nir_rs)
        cv2.imwrite(os.path.join(mask_out, out_name), mask_rs)
        count += 1

    # --- B. Process High-Res RGB scenes + RGB/General masks into 512x512 tiles ---
    rgb_dir = os.path.join(ds_root, "RGB")
    rgb_gen_dir = os.path.join(rgb_dir, "General")
    for img_path in sorted(glob.glob(os.path.join(rgb_dir, "*.jpg"))):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]

        mask_path = os.path.join(rgb_gen_dir, f"{stem}.png")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask_full = np.where(gray > 25, 1, 0).astype(np.uint8)
        if os.path.exists(mask_path):
            raw_m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if raw_m is not None:
                if raw_m.ndim == 3:
                    raw_m = raw_m[:, :, 0]
                mask_full[raw_m == 1] = 2  # IC -> Class 2
                mask_full[(raw_m == 2) | (raw_m == 3)] = 1  # Capacitors / Connectors -> Class 1

        coords = [
            (y, x)
            for y in range(0, max(1, h - tile_size + 1), stride)
            for x in range(0, max(1, w - tile_size + 1), stride)
        ]
        if not coords:
            coords = [(0, 0)]

        for t_idx, (y, x) in enumerate(coords):
            y2, x2 = min(h, y + tile_size), min(w, x + tile_size)
            t_rgb = cv2.resize(img[y:y2, x:x2], (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
            t_nir = cv2.resize(gray[y:y2, x:x2], (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
            t_mask = cv2.resize(mask_full[y:y2, x:x2], (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)

            out_name = f"rgb_{stem}_t{t_idx:02d}.png"
            cv2.imwrite(os.path.join(rgb_out, out_name), t_rgb)
            cv2.imwrite(os.path.join(nir_out, out_name), t_nir)
            cv2.imwrite(os.path.join(mask_out, out_name), t_mask)
            count += 1

    print(f"[PCB-VISION] Generated {count} harmonized {tile_size}x{tile_size} triplets in {tiles_dir}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Ingest PCB-Vision (~12.8 GB) ENVI HSI + RGB scenes.")
    parser.add_argument("--raw-dir", default="data/raw/pcb_vision")
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()

    if not args.local_only:
        download_pcb_vision(args.raw_dir)
    tile_pcb_vision(args.raw_dir)


if __name__ == "__main__":
    main()
