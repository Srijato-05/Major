import os
import sys
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

def download_spectralwaste_batch(num_samples: int = 100, target_dir: str = "data/raw/spectralwaste"):
    api = HfApi()
    repo_id = "Voxel51/SpectralWaste-Segmentation"
    print(f"Querying repository {repo_id}...")
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    rgb_files = sorted([f for f in files if f.startswith("data/rgb/") and f.endswith(".png")])
    print(f"Total available RGB frames in repo: {len(rgb_files)}")

    target_count = min(num_samples, len(rgb_files))
    selected = rgb_files[:target_count]

    for rgb_file in tqdm(selected, desc="Downloading multimodal triplets"):
        fname = os.path.basename(rgb_file)
        hsi_file = f"data/hsi/{fname}"
        mask_file = f"fields/ground_truth/mask_rgb/{fname}"

        # Target destinations
        dst_rgb = os.path.join(target_dir, "data", "rgb", fname)
        dst_hsi = os.path.join(target_dir, "data", "hsi", fname)
        dst_mask = os.path.join(target_dir, "fields", "ground_truth", "mask_rgb", fname)

        # Download if not already present
        if not os.path.exists(dst_rgb):
            hf_hub_download(repo_id=repo_id, filename=rgb_file, repo_type="dataset", local_dir=target_dir)
        if not os.path.exists(dst_hsi):
            hf_hub_download(repo_id=repo_id, filename=hsi_file, repo_type="dataset", local_dir=target_dir)
        if not os.path.exists(dst_mask):
            hf_hub_download(repo_id=repo_id, filename=mask_file, repo_type="dataset", local_dir=target_dir)

    print(f"Successfully ensured {target_count} multimodal triplets in {target_dir}!")

if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    download_spectralwaste_batch(num_samples=count)
