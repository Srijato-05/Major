import os
import json
import argparse
import cv2
import numpy as np


def _decode_coco_rle_string(rle_str: str, size_hw: tuple) -> np.ndarray:
    """Decodes COCO compressed RLE string into a binary mask (H, W) in Fortran column-major order."""
    try:
        from pycocotools import mask as mask_utils
        return mask_utils.decode({"size": list(size_hw), "counts": rle_str.encode("utf-8")})
    except ImportError:
        pass

    h, w = size_hw
    cnts = []
    p = 0
    m = 0
    n = len(rle_str)
    while p < n:
        x = 0
        k = 0
        more = 1
        while more:
            c = ord(rle_str[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = c & 0x20
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if m > 2:
            x += cnts[m - 2]
        cnts.append(x)
        m += 1

    flat = np.zeros(h * w, dtype=np.uint8)
    idx = 0
    val = 0
    for c in cnts:
        if val == 1 and c > 0:
            flat[idx:idx + c] = 1
        idx += c
        val ^= 1
    return flat.reshape((w, h)).T


def ingest_wireseg_subset(
    count: int = 1400,
    out_dir: str = "data/raw/weee_extra/wireseg_subset",
    shards: list = None,
    cache_dir: str = "data/raw/wireseg_cache",
) -> int:
    """
    Loads up to `count` samples across WireSeg-36K parquet shards (~800 scenes per shard),
    decodes `masks_rle` into Class 6 (Copper Wire Harnesses), and saves 512x512 triplets.
    """
    import pyarrow.parquet as pq

    if shards is None:
        shards = [0, 1]

    rgb_out = os.path.join(out_dir, "rgb")
    nir_out = os.path.join(out_dir, "nir")
    mask_out = os.path.join(out_dir, "masks")
    os.makedirs(rgb_out, exist_ok=True)
    os.makedirs(nir_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    saved = 0
    for shard_id in shards:
        if saved >= count:
            break
        rel_path = f"data/train-{shard_id:05d}-of-00045.parquet"
        local_parquet = os.path.join(cache_dir, rel_path)
        if not os.path.exists(local_parquet):
            from huggingface_hub import hf_hub_download
            print(f"[WIRESEG] Downloading {rel_path} into {cache_dir}...")
            local_parquet = hf_hub_download(
                "DeformX/WireSeg-36K",
                filename=rel_path,
                repo_type="dataset",
                local_dir=cache_dir,
            )

        print(f"[WIRESEG] Reading parquet shard {shard_id}: {local_parquet}")
        table = pq.read_table(local_parquet, columns=["image", "masks_rle"])
        n_rows = len(table)

        for idx in range(n_rows):
            if saved >= count:
                break
            row = table.slice(idx, 1).to_pydict()
            img_bytes = row["image"][0]["bytes"]
            nparr = np.frombuffer(img_bytes, np.uint8)
            bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if bgr is None:
                continue

            h, w = bgr.shape[:2]
            rles = json.loads(row["masks_rle"][0])
            mask = np.zeros((h, w), dtype=np.uint8)
            for rle in rles:
                m_bin = _decode_coco_rle_string(rle["counts"], tuple(rle["size"]))
                if m_bin.shape != (h, w):
                    m_bin = cv2.resize(m_bin, (w, h), interpolation=cv2.INTER_NEAREST)
                mask[m_bin > 0] = 6

            if (mask == 6).sum() < 100:
                continue

            bgr_rs = cv2.resize(bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
            mask_rs = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            nir_rs = cv2.cvtColor(bgr_rs, cv2.COLOR_BGR2GRAY)

            fname = f"wireseg_{saved:05d}.png"
            cv2.imwrite(os.path.join(rgb_out, fname), bgr_rs)
            cv2.imwrite(os.path.join(nir_out, fname), nir_rs)
            cv2.imwrite(os.path.join(mask_out, fname), mask_rs)
            saved += 1
            if saved % 150 == 0:
                print(f"[WIRESEG] Processed {saved}/{count} samples...")

    print(f"[WIRESEG] Completed: {saved} Class 6 samples -> {out_dir}")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Extract Class 6 Copper Wire subset from WireSeg-36K parquet shards.")
    parser.add_argument("--count", type=int, default=1400)
    parser.add_argument("--shards", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--out-dir", default="data/raw/weee_extra/wireseg_subset")
    parser.add_argument("--cache-dir", default="data/raw/wireseg_cache")
    args = parser.parse_args()
    ingest_wireseg_subset(count=args.count, out_dir=args.out_dir, shards=args.shards, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
