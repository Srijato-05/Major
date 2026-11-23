import os
import argparse
import urllib.request
import zipfile
import shutil
import glob
import cv2
import numpy as np

CABLE_URLS = [
    "https://github.com/m4nh/cables_dataset/archive/refs/heads/master.zip",
    "https://github.com/m4nh/cables_dataset/archive/refs/heads/main.zip",
]


def download_cables_dataset(raw_dir: str = "data/raw/cables_dataset") -> None:
    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, "cables_dataset.zip")
    if not os.path.exists(zip_path):
        last_err = None
        for url in CABLE_URLS:
            try:
                print(f"Downloading Ariadne Cable Dataset from {url}...")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as resp, open(zip_path, "wb") as out:
                    shutil.copyfileobj(resp, out)
                last_err = None
                break
            except Exception as e:
                last_err = e
        if last_err is not None:
            raise last_err

    extract_dir = os.path.join(raw_dir, "extracted")
    if not os.path.exists(extract_dir):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)


def prepare_copper_wire_triplets(
    raw_dir: str = "data/raw/cables_dataset",
    out_dir: str = "data/raw/weee_extra/copper_wires",
) -> int:
    """
    Pairs `<stem>.jpg` with `<stem>_mask_all.png` in the same directory inside `raw_dir`
    (skipping duplicate `cable_dataset_reduced` subset), converting cable pixels to Class 6.
    """
    rgb_out = os.path.join(out_dir, "rgb")
    nir_out = os.path.join(out_dir, "nir")
    mask_out = os.path.join(out_dir, "masks")
    os.makedirs(rgb_out, exist_ok=True)
    os.makedirs(nir_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    jpg_files = sorted(glob.glob(os.path.join(raw_dir, "**", "*.jpg"), recursive=True))

    count = 0
    for rgb_p in jpg_files:
        norm_p = os.path.normpath(rgb_p).lower()
        if "cable_dataset_reduced" in norm_p:
            continue

        parent = os.path.dirname(rgb_p)
        stem = os.path.splitext(os.path.basename(rgb_p))[0]
        mask_p = os.path.join(parent, f"{stem}_mask_all.png")
        if not os.path.exists(mask_p):
            continue

        img = cv2.imread(rgb_p, cv2.IMREAD_COLOR)
        m = cv2.imread(mask_p, cv2.IMREAD_UNCHANGED)
        if img is None or m is None:
            continue
        if m.ndim == 3:
            m = m[:, :, 0]
        if m.shape[:2] != img.shape[:2]:
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        mask = np.where(m > 0, 6, 0).astype(np.uint8)  # Class 6 = Copper Wire Harnesses
        if (mask == 6).sum() == 0:
            continue

        nir = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out_name = f"wire_{count:05d}.png"
        cv2.imwrite(os.path.join(rgb_out, out_name), img)
        cv2.imwrite(os.path.join(nir_out, out_name), nir)
        cv2.imwrite(os.path.join(mask_out, out_name), mask)
        count += 1

    print(f"[COPPER-WIRES] Processed {count} unique wire triplets (Class 6) -> {out_dir}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Download & ingest Ariadne Cable Dataset for Class 6 (Copper Wire Harnesses).")
    parser.add_argument("--raw-dir", default="data/raw/cables_dataset")
    parser.add_argument("--out-dir", default="data/raw/weee_extra/copper_wires")
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()

    if not args.local_only:
        download_cables_dataset(args.raw_dir)
    prepare_copper_wire_triplets(args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
