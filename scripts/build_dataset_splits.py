import os
import json
import glob
import random
import argparse
from collections import defaultdict
from typing import List, Dict, Any


def parent_group_key(rec: Dict[str, Any]) -> str:
    """
    Returns parent scene/board identifier so tiles or pairs from the same
    board/scene are kept together in a single split (zero board/scene leakage).
    """
    src = rec["source"]
    stem = rec["id"].split("/", 1)[1]
    if src == "pcb_dslr":
        return f"pcb_dslr/{stem.split('_')[0]}"  # pcb1..pcb20
    if src == "pcb_vision":
        if stem.startswith("hsi_pcb"):
            return f"pcb_vision/scene_{stem.replace('hsi_pcb', '').lstrip('0') or '0'}"
        if stem.startswith("rgb_"):
            return f"pcb_vision/scene_{stem.split('_')[1]}"
    if src == "deeppcb":
        return f"deeppcb/{stem.replace('_test', '').replace('_temp', '')}"
    return rec["id"]


def discover_triplets(raw_root: str = "data/raw", include_deeppcb: bool = False) -> List[Dict[str, Any]]:
    """Discovers all aligned multimodal (rgb, spectral, mask) triplets across raw datasets."""
    samples: List[Dict[str, Any]] = []

    # 1. SpectralWaste (852 triplets)
    sw_dir = os.path.join(raw_root, "spectralwaste")
    sw_rgb_dir = os.path.join(sw_dir, "data", "rgb")
    if os.path.exists(sw_rgb_dir):
        for fname in sorted(os.listdir(sw_rgb_dir)):
            if not fname.endswith(".png"):
                continue
            rgb_path = os.path.join(sw_rgb_dir, fname).replace("\\", "/")
            hsi_path = os.path.join(sw_dir, "data", "hsi", fname).replace("\\", "/")
            mask_path = os.path.join(sw_dir, "fields", "ground_truth", "mask_rgb", fname).replace("\\", "/")
            if os.path.exists(hsi_path) and os.path.exists(mask_path):
                samples.append({
                    "id": f"spectralwaste/{os.path.splitext(fname)[0]}",
                    "source": "spectralwaste",
                    "rgb_path": rgb_path,
                    "spectral_path": hsi_path,
                    "mask_path": mask_path,
                })

    # 2. Battery Packs (Class 3)
    bp_dir = os.path.join(raw_root, "battery_packs")
    bp_rgb_dir = os.path.join(bp_dir, "rgb")
    if os.path.exists(bp_rgb_dir):
        for fname in sorted(os.listdir(bp_rgb_dir)):
            if not fname.endswith(".jpg"):
                continue
            stem = os.path.splitext(fname)[0]
            rgb_path = os.path.join(bp_rgb_dir, fname).replace("\\", "/")
            nir_path = os.path.join(bp_dir, "nir", f"{stem}.png").replace("\\", "/")
            mask_path = os.path.join(bp_dir, "masks", f"{stem}.png").replace("\\", "/")
            if os.path.exists(nir_path) and os.path.exists(mask_path):
                samples.append({
                    "id": f"battery_packs/{stem}",
                    "source": "battery_packs",
                    "rgb_path": rgb_path,
                    "spectral_path": nir_path,
                    "mask_path": mask_path,
                })

    # 3. DeepPCB (optional; binarized B&W defect scans excluded by default when PCB-Vision/PCB-DSLR are present)
    pcb_dir = os.path.join(raw_root, "deeppcb")
    if include_deeppcb and os.path.exists(pcb_dir):
        patterns = ["**/*_test.jpg", "**/*_test.png", "**/*_temp.jpg", "**/*_temp.png"]
        candidates = []
        for pat in patterns:
            candidates.extend(glob.glob(os.path.join(pcb_dir, pat), recursive=True))
        img_paths = sorted({
            os.path.normpath(p) for p in candidates
            if os.path.sep + "masks" + os.path.sep not in os.path.normpath(p)
            and os.path.sep + "nir" + os.path.sep not in os.path.normpath(p)
        })
        for img_path in img_paths:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            nir_path = os.path.join(pcb_dir, "nir", f"{stem}.png").replace("\\", "/")
            mask_path = os.path.join(pcb_dir, "masks", f"{stem}.png").replace("\\", "/")
            if os.path.exists(nir_path) and os.path.exists(mask_path):
                samples.append({
                    "id": f"deeppcb/{stem}",
                    "source": "deeppcb",
                    "rgb_path": img_path.replace("\\", "/"),
                    "spectral_path": nir_path,
                    "mask_path": mask_path,
                })

    # 4. Standard {rgb, nir, masks} triplet directories (PCB-Vision, PCB-DSLR, weee_extra/*)
    extra_sources = [
        ("pcb_vision", os.path.join(raw_root, "pcb_vision", "tiles")),
        ("pcb_dslr", os.path.join(raw_root, "pcb_dslr", "tiles")),
    ]
    weee_extra_root = os.path.join(raw_root, "weee_extra")
    if os.path.exists(weee_extra_root):
        for sub in sorted(os.listdir(weee_extra_root)):
            sub_path = os.path.join(weee_extra_root, sub)
            if os.path.isdir(sub_path):
                extra_sources.append((f"weee_extra_{sub}", sub_path))

    for src_name, src_dir in extra_sources:
        rgb_d = os.path.join(src_dir, "rgb")
        if not os.path.exists(rgb_d):
            continue
        for fname in sorted(os.listdir(rgb_d)):
            if not (fname.endswith(".png") or fname.endswith(".jpg")):
                continue
            stem = os.path.splitext(fname)[0]
            rgb_p = os.path.join(rgb_d, fname).replace("\\", "/")
            nir_p = os.path.join(src_dir, "nir", f"{stem}.png").replace("\\", "/")
            mask_p = os.path.join(src_dir, "masks", f"{stem}.png").replace("\\", "/")
            if os.path.exists(nir_p) and os.path.exists(mask_p):
                samples.append({
                    "id": f"{src_name}/{stem}",
                    "source": src_name,
                    "rgb_path": rgb_p,
                    "spectral_path": nir_p,
                    "mask_path": mask_p,
                })

    unique_samples: List[Dict[str, Any]] = []
    seen_ids = set()
    for s in samples:
        if s["id"] not in seen_ids:
            seen_ids.add(s["id"])
            unique_samples.append(s)

    return unique_samples


def build_stratified_splits(
    raw_root: str = "data/raw",
    out_dir: str = "data/processed",
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    include_deeppcb: bool = False,
) -> Dict[str, Any]:
    """
    Splits discovered triplets deterministically (70/15/15) stratified by source
    and grouped by parent board/scene so zero scene or board leaks across splits.
    """
    samples = discover_triplets(raw_root=raw_root, include_deeppcb=include_deeppcb)

    by_source_groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for s in samples:
        by_source_groups[s["source"]][parent_group_key(s)].append(s)

    rng = random.Random(seed)
    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    per_source_summary: Dict[str, Dict[str, int]] = {}

    for source, group_map in sorted(by_source_groups.items()):
        group_keys = sorted(group_map.keys())
        rng.shuffle(group_keys)
        n_groups = len(group_keys)
        n_train_g = max(1, int(round(n_groups * train_ratio)))
        n_val_g = int(round(n_groups * val_ratio))
        if n_groups >= 3 and n_val_g == 0:
            n_val_g = 1
        if n_train_g + n_val_g >= n_groups and n_groups >= 3:
            n_train_g = max(1, n_groups - n_val_g - 1)

        tr_keys = group_keys[:n_train_g]
        va_keys = group_keys[n_train_g:n_train_g + n_val_g]
        te_keys = group_keys[n_train_g + n_val_g:]

        tr_items = [item for k in tr_keys for item in group_map[k]]
        va_items = [item for k in va_keys for item in group_map[k]]
        te_items = [item for k in te_keys for item in group_map[k]]

        splits["train"].extend(tr_items)
        splits["val"].extend(va_items)
        splits["test"].extend(te_items)
        per_source_summary[source] = {
            "total": len(tr_items) + len(va_items) + len(te_items),
            "groups": n_groups,
            "train": len(tr_items),
            "val": len(va_items),
            "test": len(te_items),
        }

    for split_name in splits:
        rng.shuffle(splits[split_name])

    os.makedirs(out_dir, exist_ok=True)
    for split_name, items in splits.items():
        out_path = os.path.join(out_dir, f"{split_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)

    summary = {
        "seed": seed,
        "include_deeppcb": include_deeppcb,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": round(1.0 - train_ratio - val_ratio, 4)},
        "totals": {k: len(v) for k, v in splits.items()},
        "total_samples": len(samples),
        "by_source": per_source_summary,
    }
    with open(os.path.join(out_dir, "split_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[SPLITS BUILT] Total={len(samples)} -> train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    for src, counts in per_source_summary.items():
        print(f"  - {src}: {counts}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build deterministic group-stratified 70/15/15 train/val/test manifests.")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-deeppcb", action="store_true", help="Include binarized DeepPCB scans")
    args = parser.parse_args()
    build_stratified_splits(
        raw_root=args.raw_root,
        out_dir=args.out_dir,
        seed=args.seed,
        include_deeppcb=args.include_deeppcb,
    )
