import os
import json
import cv2
import pytest
import torch
import numpy as np
from src.data.augmentations import IndustrialMultimodalAugmentation
from src.data.multimodal_dataset import (
    SpectralWasteDataset,
    ManifestMultimodalDataset,
    build_multimodal_dataloader,
)

def test_industrial_augmentations():
    aug = IndustrialMultimodalAugmentation(belt_speed_mps=3.0, blur_prob=1.0, glare_prob=1.0, dust_prob=1.0, cutout_prob=1.0)
    img_4ch = np.full((256, 256, 4), 128, dtype=np.uint8)
    mask = np.ones((256, 256), dtype=np.uint8)

    res = aug(img_4ch, mask)
    out_img = res["image"]
    out_mask = res["mask"]

    assert out_img.shape == (256, 256, 4)
    assert out_mask.shape == (256, 256)
    assert out_img.dtype == np.uint8

def test_spectralwaste_dataset():
    ds = SpectralWasteDataset(root_dir="data/raw/spectralwaste", img_size=(320, 320), is_training=False)
    assert len(ds) > 0

    item = ds[0]
    assert "image" in item
    assert "mask" in item
    assert item["image"].shape == (4, 320, 320)
    assert item["mask"].shape == (320, 320)
    assert item["image"].dtype == torch.float32
    assert item["mask"].dtype == torch.int64

def test_multimodal_dataloader():
    loader = build_multimodal_dataloader(root_dir="data/raw/spectralwaste", batch_size=2, img_size=(320, 320), is_training=False)
    batch = next(iter(loader))

    assert batch["image"].shape == (2, 4, 320, 320)
    assert batch["mask"].shape == (2, 320, 320)


def test_stratified_splits_zero_overlap():
    from scripts.build_dataset_splits import parent_group_key

    splits = {}
    groups = {}
    for name in ["train", "val", "test"]:
        path = os.path.join("data", "processed", f"{name}.json")
        assert os.path.exists(path), f"Missing split manifest: {path}"
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        assert len(records) > 0
        splits[name] = {r["id"] for r in records}
        groups[name] = {parent_group_key(r) for r in records}

    assert splits["train"].isdisjoint(splits["val"])
    assert splits["train"].isdisjoint(splits["test"])
    assert splits["val"].isdisjoint(splits["test"])

    # Verify zero parent board/scene leakage across splits
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])


def test_manifest_dataset_and_no_battery_nir_leakage():
    ds = ManifestMultimodalDataset(manifest_path="data/processed/val.json", img_size=(320, 320), is_training=False)
    assert len(ds) > 0
    sample = ds[0]
    assert sample["image"].shape == (4, 320, 320)
    assert sample["image"].dtype == torch.float32
    assert 0.0 <= float(sample["image"].min()) and float(sample["image"].max()) <= 1.0
    assert sample["mask"].shape == (320, 320)
    assert sample["mask"].dtype == torch.int64
    assert 0 <= int(sample["mask"].min()) and int(sample["mask"].max()) <= 6

    # Verify battery pack nir matches grayscale rgb (no +60 mask leakage)
    rgb = cv2.imread("data/raw/battery_packs/rgb/battery_0000.jpg", cv2.IMREAD_COLOR)
    nir = cv2.imread("data/raw/battery_packs/nir/battery_0000.png", cv2.IMREAD_UNCHANGED)
    mask = cv2.imread("data/raw/battery_packs/masks/battery_0000.png", cv2.IMREAD_UNCHANGED)
    expected_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    assert np.array_equal(nir, expected_gray)
    assert set(np.unique(mask).tolist()).issubset({0, 3})


def test_coco_weee_importer(tmp_path):
    from scripts.ingest_coco_weee import ingest_coco_dir

    in_dir = tmp_path / "coco_in"
    out_dir = tmp_path / "coco_out"
    in_dir.mkdir()

    img = np.full((64, 64, 3), 100, dtype=np.uint8)
    cv2.imwrite(str(in_dir / "sample.png"), img)
    coco = {
        "images": [{"id": 1, "file_name": "sample.png"}],
        "annotations": [{"id": 1, "image_id": 1, "segmentation": [[10, 10, 40, 10, 40, 40, 10, 40]]}],
    }
    (in_dir / "annotations.json").write_text(json.dumps(coco), encoding="utf-8")

    count = ingest_coco_dir(str(in_dir), str(out_dir), class_id=5)
    assert count == 1
    mask = cv2.imread(str(list((out_dir / "masks").glob("*.png"))[0]), cv2.IMREAD_UNCHANGED)
    nir = cv2.imread(str(list((out_dir / "nir").glob("*.png"))[0]), cv2.IMREAD_UNCHANGED)
    assert 5 in np.unique(mask)
    assert np.all(nir == 100)


def test_yolo_selective_class_map_skips_unmapped(tmp_path):
    from scripts.ingest_coco_weee import ingest_yolo_dir, parse_class_map

    in_dir = tmp_path / "yolo_in"
    out_dir = tmp_path / "yolo_out"
    (in_dir / "images").mkdir(parents=True)
    (in_dir / "labels").mkdir(parents=True)

    # Image 1 has class 0 (Battery -> 3) and class 6 (tire -> unmapped, ignored)
    cv2.imwrite(str(in_dir / "images" / "keep.jpg"), np.full((64, 64, 3), 120, dtype=np.uint8))
    (in_dir / "labels" / "keep.txt").write_text("0 0.5 0.5 0.4 0.4\n6 0.2 0.2 0.2 0.2\n", encoding="utf-8")

    # Image 2 only has class 3 (glass bottle -> unmapped, whole image skipped)
    cv2.imwrite(str(in_dir / "images" / "skip.jpg"), np.full((64, 64, 3), 90, dtype=np.uint8))
    (in_dir / "labels" / "skip.txt").write_text("3 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    cmap = parse_class_map("0:3,1:5")
    count = ingest_yolo_dir(str(in_dir), str(out_dir), class_map=cmap)
    assert count == 1
    mask = cv2.imread(str(list((out_dir / "masks").glob("*.png"))[0]), cv2.IMREAD_UNCHANGED)
    assert set(np.unique(mask).tolist()).issubset({0, 3})
