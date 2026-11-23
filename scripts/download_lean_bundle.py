import os
import sys
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.ingest_deeppcb import download_and_extract_deeppcb, generate_pcb_masks_and_nir
from scripts.ingest_pcb_vision import download_pcb_vision, tile_pcb_vision
from scripts.ingest_pcb_dslr import download_pcb_dslr, prepare_pcb_dslr_tiles
from scripts.build_dataset_splits import build_stratified_splits


def main():
    parser = argparse.ArgumentParser(description="Download and harmonize the Lean ~20 GB multi-dataset WEEE bundle.")
    parser.add_argument("--skip-deeppcb", action="store_true")
    parser.add_argument("--skip-pcb-vision", action="store_true")
    parser.add_argument("--skip-pcb-dslr", action="store_true")
    parser.add_argument("--local-only", action="store_true", help="Skip network downloads and process local archives/files only")
    args = parser.parse_args()

    print("=" * 75)
    print("[LEAN BUNDLE] STARTING FULL DATASET INGESTION (~17-20 GB)")
    print("=" * 75)

    if not args.skip_deeppcb:
        try:
            if not args.local_only:
                download_and_extract_deeppcb("data/raw/deeppcb")
            generate_pcb_masks_and_nir("data/raw/deeppcb")
        except Exception as e:
            print(f"[WARNING] DeepPCB step failed: {e}. Continuing with remaining datasets...")

    if not args.skip_pcb_vision:
        try:
            if not args.local_only:
                download_pcb_vision("data/raw/pcb_vision")
            tile_pcb_vision("data/raw/pcb_vision")
        except Exception as e:
            print(f"[WARNING] PCB-Vision step failed: {e}. Continuing with remaining datasets...")

    if not args.skip_pcb_dslr:
        try:
            if not args.local_only:
                download_pcb_dslr("data/raw/pcb_dslr")
            prepare_pcb_dslr_tiles("data/raw/pcb_dslr")
        except Exception as e:
            print(f"[WARNING] PCB-DSLR step failed: {e}. Continuing with remaining datasets...")

    print("\nRebuilding deterministic stratified 70/15/15 splits across all discovered corpora...")
    build_stratified_splits(raw_root="data/raw", out_dir="data/processed", seed=42)
    print("[LEAN BUNDLE COMPLETE]")


if __name__ == "__main__":
    main()

