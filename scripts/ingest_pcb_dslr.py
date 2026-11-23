import os
import argparse
import json
import urllib.request
import zipfile
import shutil
import glob
import cv2
import numpy as np

ZENODO_SEARCH_URL = "https://zenodo.org/api/records?q=%22PCB-DSLR%22"


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


def download_pcb_dslr(raw_dir: str = "data/raw/pcb_dslr") -> None:
    os.makedirs(raw_dir, exist_ok=True)
    req = urllib.request.Request(ZENODO_SEARCH_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    hits = data.get("hits", {}).get("hits", [])
    if not hits:
        raise RuntimeError("Could not locate PCB-DSLR record on Zenodo.")

    record = hits[0]
    files = record.get("files", [])
    print(f"[PCB-DSLR] Resolved Zenodo record {record.get('id')} with {len(files)} file(s).")
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


def _rasterize_ic_annotations(mask: np.ndarray, annot_path: str) -> None:
    """Parses PCB-DSLR rotated bounding boxes `cx cy w h angle [label]` into Class 2 (IC Molds)."""
    if not os.path.exists(annot_path):
        return
    with open(annot_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cx, cy, bw, bh, angle = [float(x) for x in parts[:5]]
            except ValueError:
                continue
            box = cv2.boxPoints(((cx, cy), (bw, bh), angle))
            cv2.fillPoly(mask, [np.int32(np.round(box))], 2)  # Class 2 = IC Molds & Chips


def prepare_pcb_dslr_tiles(raw_dir: str = "data/raw/pcb_dslr", target_size: int = 640) -> int:
    """
    Converts PCB-DSLR `recX.jpg`, `recX-mask.png` (Class 1 PCB), and `recX-annot.txt` (Class 2 ICs)
    into full-board views plus 2x2 high-resolution crops under `tiles/{rgb,nir,masks}`.
    """
    tiles_dir = os.path.join(raw_dir, "tiles")
    rgb_out = os.path.join(tiles_dir, "rgb")
    nir_out = os.path.join(tiles_dir, "nir")
    mask_out = os.path.join(tiles_dir, "masks")
    os.makedirs(rgb_out, exist_ok=True)
    os.makedirs(nir_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    candidates = sorted(
        glob.glob(os.path.join(raw_dir, "**", "*.jpg"), recursive=True)
        + glob.glob(os.path.join(raw_dir, "**", "*.png"), recursive=True)
    )
    img_files = [
        p for p in candidates
        if os.path.sep + "tiles" + os.path.sep not in os.path.normpath(p)
        and "-mask" not in os.path.basename(p).lower()
        and "mask" not in os.path.basename(p).lower()
    ]

    count = 0
    for img_path in img_files:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        parent = os.path.dirname(img_path)
        stem = os.path.splitext(os.path.basename(img_path))[0]
        board_tag = os.path.basename(parent)

        mask_path = os.path.join(parent, f"{stem}-mask.png")
        annot_path = os.path.join(parent, f"{stem}-annot.txt")

        if os.path.exists(mask_path):
            m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if m is not None and m.ndim == 3:
                m = m[:, :, 0]
            mask = np.where(m > 0, 1, 0).astype(np.uint8) if m is not None else np.ones((h, w), dtype=np.uint8)
        else:
            mask = np.ones((h, w), dtype=np.uint8)

        _rasterize_ic_annotations(mask, annot_path)
        nir = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Save full-board view + 2x2 high-res quadrants where foreground exists
        regions = [("full", (0, h, 0, w))]
        mid_y, mid_x = h // 2, w // 2
        regions.extend([
            ("q0", (0, mid_y, 0, mid_x)),
            ("q1", (0, mid_y, mid_x, w)),
            ("q2", (mid_y, h, 0, mid_x)),
            ("q3", (mid_y, h, mid_x, w)),
        ])

        for tag, (y1, y2, x1, x2) in regions:
            sub_mask = mask[y1:y2, x1:x2]
            if tag != "full" and (sub_mask > 0).mean() < 0.02:
                continue
            rgb_rs = cv2.resize(img[y1:y2, x1:x2], (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            nir_rs = cv2.resize(nir[y1:y2, x1:x2], (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            mask_rs = cv2.resize(sub_mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

            out_name = f"{board_tag}_{stem}_{tag}.png"
            cv2.imwrite(os.path.join(rgb_out, out_name), rgb_rs)
            cv2.imwrite(os.path.join(nir_out, out_name), nir_rs)
            cv2.imwrite(os.path.join(mask_out, out_name), mask_rs)
            count += 1

    print(f"[PCB-DSLR] Processed {count} triplets (Class 1 PCB + Class 2 ICs) into {tiles_dir}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Ingest PCB-DSLR (Class 1 PCB + Class 2 ICs).")
    parser.add_argument("--raw-dir", default="data/raw/pcb_dslr")
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()

    if not args.local_only:
        download_pcb_dslr(args.raw_dir)
    prepare_pcb_dslr_tiles(args.raw_dir)


if __name__ == "__main__":
    main()
