import os
import json
import glob
import argparse
import cv2
import numpy as np
from typing import Dict, List, Optional


def parse_class_map(spec: Optional[str]) -> Optional[Dict[int, int]]:
    """Parses a mapping string like '0:3,1:5' into {0: 3, 1: 5}."""
    if not spec:
        return None
    mapping: Dict[int, int] = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        src_s, dst_s = pair.split(":")
        mapping[int(src_s.strip())] = int(dst_s.strip())
    return mapping


def _refine_bbox_grabcut(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Optional[np.ndarray]:
    """
    Runs GrabCut (2 iters) on a bounding box region to tighten rectangular labels onto the object.
    Returns a boolean mask of shape (y2 - y1, x2 - x1) if foreground area is 20%-95% of the box;
    otherwise returns None so caller falls back to the full rectangle.
    """
    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(x1, min(w, int(x2)))
    y2 = max(y1, min(h, int(y2)))

    bw = x2 - x1
    bh = y2 - y1
    box_area = bw * bh
    if bw < 16 or bh < 16 or box_area < 256:
        return None

    try:
        pad = max(8, int(0.15 * max(bw, bh)))
        ox1 = max(0, x1 - pad)
        oy1 = max(0, y1 - pad)
        ox2 = min(w, x2 + pad)
        oy2 = min(h, y2 + pad)
        roi = img[oy1:oy2, ox1:ox2]
        roi_h, roi_w = roi.shape[:2]

        rx1 = max(1, x1 - ox1)
        ry1 = max(1, y1 - oy1)
        rx2 = min(roi_w - 1, x2 - ox1)
        ry2 = min(roi_h - 1, y2 - oy1)
        if rx2 - rx1 < 8 or ry2 - ry1 < 8:
            return None

        gc_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        rect = (rx1, ry1, rx2 - rx1, ry2 - ry1)

        max_side = max(roi_h, roi_w)
        if max_side > 160:
            scale = 160.0 / float(max_side)
            sw = max(16, int(round(roi_w * scale)))
            sh = max(16, int(round(roi_h * scale)))
            s_roi = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)
            srx1 = max(1, int(round(rx1 * scale)))
            sry1 = max(1, int(round(ry1 * scale)))
            srx2 = min(sw - 1, max(srx1 + 4, int(round(rx2 * scale))))
            sry2 = min(sh - 1, max(sry1 + 4, int(round(ry2 * scale))))
            s_mask = np.zeros((sh, sw), dtype=np.uint8)
            cv2.grabCut(s_roi, s_mask, (srx1, sry1, srx2 - srx1, sry2 - sry1), bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_RECT)
            s_fg = ((s_mask == cv2.GC_FGD) | (s_mask == cv2.GC_PR_FGD)).astype(np.uint8)
            fg_roi = cv2.resize(s_fg, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            cv2.grabCut(roi, gc_mask, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_RECT)
            fg_roi = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)

        fg_box = fg_roi[(y1 - oy1):(y2 - oy1), (x1 - ox1):(x2 - ox1)]
        if fg_box.shape != (bh, bw):
            return None

        ratio = float(fg_box.sum()) / float(box_area)
        if 0.20 <= ratio <= 0.95:
            return fg_box
    except Exception:
        return None
    return None


def _draw_bbox_or_grabcut(img: np.ndarray, mask: np.ndarray, x1: int, y1: int, x2: int, y2: int, target_cid: int, refine_bbox: bool):
    """Fills a bounding box into `mask`, optionally refined with GrabCut foreground estimation."""
    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(x1, min(w, int(x2)))
    y2 = max(y1, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return
    if refine_bbox:
        refined = _refine_bbox_grabcut(img, x1, y1, x2, y2)
        if refined is not None:
            patch = mask[y1:y2, x1:x2]
            patch[refined] = target_cid
            return
    cv2.rectangle(mask, (x1, y1), (x2, y2), target_cid, -1)


def ingest_coco_dir(
    input_dir: str,
    out_dir: str,
    class_id: Optional[int] = None,
    class_map: Optional[Dict[int, int]] = None,
    refine_bbox: bool = False,
) -> int:
    """
    Converts COCO JSON polygon/bbox annotations into (rgb, nir, mask) triplets.
    When `class_map` is provided, only annotations whose category_id is in `class_map`
    are rasterized, and images with zero mapped annotations are skipped.
    """
    ann_files = glob.glob(os.path.join(input_dir, "**", "*annotations*.json"), recursive=True) + \
                glob.glob(os.path.join(input_dir, "**", "*.coco.json"), recursive=True)
    if not ann_files:
        ann_files = glob.glob(os.path.join(input_dir, "**", "*.json"), recursive=True)

    rgb_out = os.path.join(out_dir, "rgb")
    nir_out = os.path.join(out_dir, "nir")
    mask_out = os.path.join(out_dir, "masks")
    os.makedirs(rgb_out, exist_ok=True)
    os.makedirs(nir_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    count = 0
    for ann_path in sorted(set(ann_files)):
        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)
        if "images" not in coco or "annotations" not in coco:
            continue

        ann_dir = os.path.dirname(ann_path)
        anns_by_img: Dict[int, List[dict]] = {}
        for ann in coco["annotations"]:
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        for img_meta in coco["images"]:
            fname = img_meta["file_name"]
            candidates = [
                os.path.join(ann_dir, fname),
                os.path.join(input_dir, fname),
                os.path.join(input_dir, "images", os.path.basename(fname)),
            ]
            img_path = next((p for p in candidates if os.path.exists(p)), None)
            if img_path is None:
                continue

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            mapped_hits = 0

            for ann in anns_by_img.get(img_meta["id"], []):
                if class_map is not None:
                    src_cid = int(ann.get("category_id", -1))
                    if src_cid not in class_map:
                        continue
                    target_cid = int(class_map[src_cid])
                else:
                    if class_id is None:
                        continue
                    target_cid = int(class_id)

                segs = ann.get("segmentation", [])
                drew = False
                if isinstance(segs, list) and segs:
                    for poly in segs:
                        if isinstance(poly, list) and len(poly) >= 6:
                            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                            cv2.fillPoly(mask, [pts], target_cid)
                            drew = True
                elif "bbox" in ann:
                    bx, by, bw, bh = [int(round(v)) for v in ann["bbox"]]
                    _draw_bbox_or_grabcut(img, mask, bx, by, bx + bw, by + bh, target_cid, refine_bbox=refine_bbox)
                    drew = True
                if drew:
                    mapped_hits += 1

            if class_map is not None and mapped_hits == 0:
                continue

            nir = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            stem = f"{os.path.splitext(os.path.basename(fname))[0]}_{count:05d}"
            cv2.imwrite(os.path.join(rgb_out, f"{stem}.png"), img)
            cv2.imwrite(os.path.join(nir_out, f"{stem}.png"), nir)
            cv2.imwrite(os.path.join(mask_out, f"{stem}.png"), mask)
            count += 1
    return count


def ingest_yolo_dir(
    input_dir: str,
    out_dir: str,
    class_id: Optional[int] = None,
    class_map: Optional[Dict[int, int]] = None,
    refine_bbox: bool = False,
) -> int:
    """
    Converts YOLO polygon/bbox .txt annotations into (rgb, nir, mask) triplets.
    When `class_map` is provided, only annotations whose source class is in `class_map`
    are rasterized, and images with zero mapped objects are skipped.
    """
    rgb_out = os.path.join(out_dir, "rgb")
    nir_out = os.path.join(out_dir, "nir")
    mask_out = os.path.join(out_dir, "masks")
    os.makedirs(rgb_out, exist_ok=True)
    os.makedirs(nir_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    img_files = sorted(
        glob.glob(os.path.join(input_dir, "**", "*.jpg"), recursive=True)
        + glob.glob(os.path.join(input_dir, "**", "*.png"), recursive=True)
    )
    count = 0
    for img_path in img_files:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        parent = os.path.dirname(img_path)
        lbl_candidates = [
            os.path.join(parent, f"{stem}.txt"),
            os.path.join(os.path.dirname(parent), "labels", f"{stem}.txt"),
        ]
        lbl_path = next((p for p in lbl_candidates if os.path.exists(p)), None)
        if lbl_path is None:
            continue

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mapped_hits = 0

        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = [float(x) for x in line.strip().split()]
                if len(parts) < 5:
                    continue
                src_cid = int(parts[0])
                if class_map is not None:
                    if src_cid not in class_map:
                        continue
                    target_cid = int(class_map[src_cid])
                else:
                    if class_id is None:
                        continue
                    target_cid = int(class_id)

                if len(parts) == 5:
                    _, cx, cy, bw, bh = parts
                    x1 = max(0, int((cx - bw / 2.0) * w))
                    y1 = max(0, int((cy - bh / 2.0) * h))
                    x2 = min(w, int((cx + bw / 2.0) * w))
                    y2 = min(h, int((cy + bh / 2.0) * h))
                    _draw_bbox_or_grabcut(img, mask, x1, y1, x2, y2, target_cid, refine_bbox=refine_bbox)
                    mapped_hits += 1
                elif len(parts) >= 7:
                    coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
                    coords[:, 0] *= w
                    coords[:, 1] *= h
                    cv2.fillPoly(mask, [coords.astype(np.int32)], target_cid)
                    mapped_hits += 1

        if class_map is not None and mapped_hits == 0:
            continue

        nir = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out_stem = f"{stem}_{count:05d}"
        cv2.imwrite(os.path.join(rgb_out, f"{out_stem}.png"), img)
        cv2.imwrite(os.path.join(nir_out, f"{out_stem}.png"), nir)
        cv2.imwrite(os.path.join(mask_out, f"{out_stem}.png"), mask)
        count += 1
    return count


def ingest_external_weee(
    input_dir: str,
    name: str,
    class_id: Optional[int] = None,
    class_map: Optional[Dict[int, int]] = None,
    fmt: str = "auto",
    refine_bbox: bool = False,
) -> int:
    if class_id is None and class_map is None:
        raise ValueError("Provide either --class-id or --class-map.")

    out_dir = os.path.join("data", "raw", "weee_extra", name)
    if fmt == "auto":
        has_json = bool(glob.glob(os.path.join(input_dir, "**", "*.json"), recursive=True))
        fmt = "coco" if has_json else "yolo"

    if fmt == "coco":
        count = ingest_coco_dir(input_dir, out_dir, class_id=class_id, class_map=class_map, refine_bbox=refine_bbox)
    elif fmt == "yolo":
        count = ingest_yolo_dir(input_dir, out_dir, class_id=class_id, class_map=class_map, refine_bbox=refine_bbox)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    print(f"[WEEE-IMPORT:{name}] Converted {count} samples (class_id={class_id}, class_map={class_map}, fmt={fmt}, refine_bbox={refine_bbox}) -> {out_dir}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Convert Roboflow/Kaggle COCO or YOLO exports into 7-class triplets.")
    parser.add_argument("--input-dir", required=True, help="Extracted COCO or YOLO dataset directory")
    parser.add_argument("--class-id", type=int, choices=[1, 2, 3, 4, 5, 6], default=None, help="Single target 7-class foreground ID")
    parser.add_argument("--class-map", type=str, default=None, help="Selective mapping e.g. '0:3,1:5' (skips unmapped classes & empty images)")
    parser.add_argument("--name", required=True, help="Output folder name under data/raw/weee_extra/")
    parser.add_argument("--format", choices=["auto", "coco", "yolo"], default="auto")
    parser.add_argument("--refine-bbox", action="store_true", help="Tighten rectangular bounding-box masks with GrabCut foreground estimation")
    args = parser.parse_args()

    cmap = parse_class_map(args.class_map)
    ingest_external_weee(args.input_dir, args.name, class_id=args.class_id, class_map=cmap, fmt=args.format, refine_bbox=args.refine_bbox)


if __name__ == "__main__":
    main()
