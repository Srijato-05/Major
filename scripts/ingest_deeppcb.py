import os
import argparse
import urllib.request
import zipfile
import shutil
import glob
import cv2
import numpy as np

DEEPPCB_URL = "https://github.com/tangsanli5201/DeepPCB/archive/refs/heads/master.zip"


def download_and_extract_deeppcb(raw_dir: str = "data/raw/deeppcb") -> None:
    """Downloads and extracts the full DeepPCB dataset into raw_dir."""
    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, "DeepPCB-master.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading DeepPCB dataset from {DEEPPCB_URL}...")
        req = urllib.request.Request(DEEPPCB_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(zip_path, "wb") as out:
            shutil.copyfileobj(resp, out)
    extract_dir = os.path.join(raw_dir, "extracted")
    if not os.path.exists(extract_dir):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)


def generate_pcb_masks_and_nir(raw_dir: str = "data/raw/deeppcb") -> int:
    """
    Discovers *_test.* / *_temp.* images inside raw_dir and generates
    Class 1 (High-Grade PCB) segmentation masks and mask-independent grayscale NIR bands.
    """
    masks_dir = os.path.join(raw_dir, "masks")
    nir_dir = os.path.join(raw_dir, "nir")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(nir_dir, exist_ok=True)

    patterns = ["**/*_test.jpg", "**/*_test.png", "**/*_temp.jpg", "**/*_temp.png"]
    candidate_files = []
    for pat in patterns:
        candidate_files.extend(glob.glob(os.path.join(raw_dir, pat), recursive=True))

    # Deduplicate and exclude output directories
    img_files = sorted({
        os.path.normpath(p) for p in candidate_files
        if os.path.sep + "masks" + os.path.sep not in os.path.normpath(p)
        and os.path.sep + "nir" + os.path.sep not in os.path.normpath(p)
    })

    processed = 0
    for img_path in img_files:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # DeepPCB tiles depict PCB board substrate & copper traces -> Class 1 (High-Grade PCB)
        # Keep non-trivial foreground regions (or full PCB board region)
        mask = np.ones((h, w), dtype=np.uint8)  # Class 1 = High-Grade PCB
        nir_band = gray.copy()

        stem = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(os.path.join(masks_dir, f"{stem}.png"), mask)
        cv2.imwrite(os.path.join(nir_dir, f"{stem}.png"), nir_band)
        processed += 1

    print(f"[DEEPPCB] Processed {processed} PCB images into Class 1 masks & grayscale NIR in {raw_dir}")
    return processed


def main():
    parser = argparse.ArgumentParser(description="Ingest and preprocess DeepPCB dataset for Class 1 (High-Grade PCB).")
    parser.add_argument("--raw-dir", default="data/raw/deeppcb", help="DeepPCB root directory")
    parser.add_argument("--local-only", action="store_true", help="Only process already-downloaded files in raw-dir")
    args = parser.parse_args()

    if not args.local_only:
        download_and_extract_deeppcb(args.raw_dir)
    generate_pcb_masks_and_nir(args.raw_dir)


if __name__ == "__main__":
    main()

