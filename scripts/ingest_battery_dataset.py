import os
import sys
import argparse
import urllib.request
import json
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def download_file(url, out_path):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        return True
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read()
            with open(out_path, 'wb') as f:
                f.write(content)
        return True
    except Exception:
        return False

def process_local_battery_corpus(raw_dir="data/raw/battery_packs"):
    """
    Generates Class 3 (Battery Pack Units) segmentation masks and mask-independent
    grayscale luminance 4th-channel images from RGB scans in raw_dir/rgb.
    """
    os.makedirs(os.path.join(raw_dir, "nir"), exist_ok=True)
    os.makedirs(os.path.join(raw_dir, "masks"), exist_ok=True)

    print("\nGenerating mask-independent 4th channel & Class 3 (Battery Pack) segmentation masks...")
    processed_count = 0

    for f in sorted(os.listdir(os.path.join(raw_dir, "rgb"))):
        if not f.endswith(".jpg"):
            continue
        base_id = os.path.splitext(f)[0]
        rgb_path = os.path.join(raw_dir, "rgb", f)
        nir_path = os.path.join(raw_dir, "nir", f"{base_id}.png")
        mask_path = os.path.join(raw_dir, "masks", f"{base_id}.png")

        img = cv2.imread(rgb_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Foreground battery extraction via Otsu & adaptive edge thresholding
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Keep largest connected components (the battery body)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros((h, w), dtype=np.uint8)

        if cnts:
            cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
            for c in cnts_sorted[:3]:
                if cv2.contourArea(c) > (h * w * 0.05):
                    cv2.drawContours(mask, [c], -1, 3, -1)  # Class 3 = Battery Pack Units

        if (mask == 3).sum() < (h * w * 0.04):
            # Fallback: Central bounding elliptical region if contrast is low
            center_x, center_y = w // 2, h // 2
            axes = (int(w * 0.35), int(h * 0.35))
            cv2.ellipse(mask, (center_x, center_y), axes, 0, 0, 360, 3, -1)

        # Mask-independent 4th channel (grayscale luminance) to avoid ground-truth leakage
        nir_band = gray.copy()

        cv2.imwrite(nir_path, nir_band)
        cv2.imwrite(mask_path, mask)
        processed_count += 1

    print(f"[PREPROCESSING COMPLETE] Generated {processed_count} synchronized 4-channel battery triplets.")
    print("=" * 70)
    return processed_count

def ingest_battery_corpus(target_count=550, rebuild_local=False):
    print("=" * 70)
    if rebuild_local:
        print("[DATASET INGESTION] REBUILDING LOCAL BATTERY PACK MASKS & 4TH CHANNEL")
    else:
        print(f"[DATASET INGESTION] DOWNLOADING {target_count}+ REAL BATTERY PACK SAMPLES")
    print("=" * 70)

    raw_dir = "data/raw/battery_packs"
    os.makedirs(os.path.join(raw_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(raw_dir, "nir"), exist_ok=True)
    os.makedirs(os.path.join(raw_dir, "masks"), exist_ok=True)

    if not rebuild_local:
        api_url = "https://huggingface.co/api/datasets/kdkd1/waste-garbage-management-dataset/tree/main/battery"
        req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            tree = json.loads(resp.read().decode())

        battery_items = [item['path'] for item in tree if item['path'].endswith('.jpg')]
        print(f"Discovered {len(battery_items)} candidate battery images on remote repo.")
        selected_items = battery_items[:target_count]
        print(f"Targeting download of {len(selected_items)} images...")

        base_url = "https://huggingface.co/datasets/kdkd1/waste-garbage-management-dataset/resolve/main/"
        download_tasks = []

        with ThreadPoolExecutor(max_workers=16) as executor:
            for idx, item_path in enumerate(selected_items):
                file_name = f"battery_{idx:04d}.jpg"
                out_file = os.path.join(raw_dir, "rgb", file_name)
                remote_url = base_url + item_path
                download_tasks.append((executor.submit(download_file, remote_url, out_file), file_name, out_file))

            completed = 0
            for fut, fname, fpath in download_tasks:
                if fut.result():
                    completed += 1
                if completed % 100 == 0 or completed == len(selected_items):
                    print(f"Downloaded {completed}/{len(selected_items)} battery RGB images...")

        print(f"\n[DOWNLOAD COMPLETE] Successfully downloaded {completed} raw battery images.")

    process_local_battery_corpus(raw_dir=raw_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and preprocess Battery Pack dataset.")
    parser.add_argument("count", nargs="?", type=int, default=550, help="Number of samples to download")
    parser.add_argument("--rebuild-local", action="store_true", help="Regenerate masks/nir from existing local rgb images without network")
    args = parser.parse_args()
    ingest_battery_corpus(target_count=args.count, rebuild_local=args.rebuild_local)
