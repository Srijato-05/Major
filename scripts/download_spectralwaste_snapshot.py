import os
import sys
from huggingface_hub import snapshot_download

def download_full_spectralwaste():
    print("=" * 60)
    print("DOWNLOADING FULL SPECTRALWASTE (852 TRIPLETS) VIA SNAPSHOT")
    print("=" * 60)
    
    target_dir = os.path.abspath("data/raw/spectralwaste")
    
    # We only include rgb, hsi, and mask folders, omitting the heavy 3D hyper-cubes
    allow_patterns = [
        "data/rgb/*",
        "data/hsi/*",
        "fields/ground_truth/mask_rgb/*",
        "categories.json"
    ]
    
    print(f"Target Directory: {target_dir}")
    print(f"Allow patterns: {allow_patterns}")
    
    path = snapshot_download(
        repo_id="Voxel51/SpectralWaste-Segmentation",
        repo_type="dataset",
        local_dir=target_dir,
        allow_patterns=allow_patterns,
        max_workers=8
    )
    
    print(f"\n[SUCCESS] Snapshot download complete at: {path}")
    
    # Check counts
    rgb_p = os.path.join(target_dir, "data", "rgb")
    if os.path.exists(rgb_p):
        print(f"Total RGB frames present: {len(os.listdir(rgb_p))}")

if __name__ == "__main__":
    download_full_spectralwaste()
