import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Tuple, Optional, List
from src.data.augmentations import IndustrialMultimodalAugmentation

class SpectralWasteDataset(Dataset):
    """
    Multimodal Dataset Loader for SpectralWaste benchmark.
    Fuses high-spatial RGB (3 channels) with synchronized SWIR/HSI (1 channel)
    to form standardized 4-channel tensors: (4, H, W).
    """

    def __init__(
        self,
        root_dir: str = "data/raw/spectralwaste",
        img_size: Tuple[int, int] = (640, 640),
        is_training: bool = True,
        augmentation: Optional[IndustrialMultimodalAugmentation] = None,
        class_mapping: Optional[Dict[int, int]] = None
    ):
        self.root_dir = root_dir
        self.img_size = img_size
        self.is_training = is_training
        self.augmentation = augmentation
        if class_mapping is None:
            from src.utils.config_loader import CONFIG
            class_mapping = CONFIG.get("taxonomy", {}).get("dataset_mappings", {}).get("spectralwaste")
        self.class_mapping = class_mapping

        # Discover all available triplets
        rgb_dir = os.path.join(root_dir, "data", "rgb")
        self.sample_names = []
        if os.path.exists(rgb_dir):
            for f in sorted(os.listdir(rgb_dir)):
                if f.endswith(".png"):
                    base_name = f
                    hsi_path = os.path.join(root_dir, "data", "hsi", base_name)
                    mask_path = os.path.join(root_dir, "fields", "ground_truth", "mask_rgb", base_name)
                    if os.path.exists(hsi_path) and os.path.exists(mask_path):
                        self.sample_names.append(base_name)

    def __len__(self) -> int:
        return len(self.sample_names)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        base_name = self.sample_names[idx]
        rgb_path = os.path.join(self.root_dir, "data", "rgb", base_name)
        hsi_path = os.path.join(self.root_dir, "data", "hsi", base_name)
        mask_path = os.path.join(self.root_dir, "fields", "ground_truth", "mask_rgb", base_name)

        # 1. Load RGB image (BGR -> RGB)
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # 2. Load HSI/SWIR spectral band
        hsi = cv2.imread(hsi_path, cv2.IMREAD_UNCHANGED)
        if len(hsi.shape) == 3:
            # If 3-channel pseudo-color, take the first channel or mean as single spectral band
            spectral_band = hsi[:, :, 0:1]
        else:
            spectral_band = hsi[:, :, np.newaxis]

        # 3. Stack into 4-channel numpy array: (H, W, 4)
        image_4ch = np.concatenate([rgb, spectral_band], axis=2)

        # 4. Load segmentation mask
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        # Remap raw dataset class indices to system taxonomy classes if configured
        if self.class_mapping is not None:
            remapped_mask = np.zeros_like(mask)
            for raw_id, target_id in self.class_mapping.items():
                remapped_mask[mask == int(raw_id)] = int(target_id)
            mask = remapped_mask

        # 5. Apply Industrial Conveyor Augmentations
        if self.is_training and self.augmentation is not None:
            aug_res = self.augmentation(image_4ch, mask)
            image_4ch = aug_res["image"]
            mask = aug_res["mask"]

        # 6. Resize to standard model dimension (img_size)
        image_4ch = cv2.resize(image_4ch, self.img_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        # 7. Convert to PyTorch tensors
        # Image: [H, W, 4] -> [4, H, W], normalized [0.0, 1.0]
        tensor_image = torch.from_numpy(image_4ch).permute(2, 0, 1).float() / 255.0
        tensor_mask = torch.from_numpy(mask).long()

        return {
            "image": tensor_image,  # [4, H, W]
            "mask": tensor_mask,    # [H, W]
            "filename": base_name
        }


class BatteryPackDataset(Dataset):
    """
    Multimodal Dataset Loader for Class 2 (Battery Pack Units).
    Fuses high-resolution RGB line scans with simulated metallic SWIR response (4 channels).
    """

    def __init__(
        self,
        root_dir: str = "data/raw/battery_packs",
        img_size: Tuple[int, int] = (640, 640),
        is_training: bool = True,
        augmentation: Optional[IndustrialMultimodalAugmentation] = None
    ):
        self.root_dir = root_dir
        self.img_size = img_size
        self.is_training = is_training
        self.augmentation = augmentation

        rgb_dir = os.path.join(root_dir, "rgb")
        self.sample_names = []
        if os.path.exists(rgb_dir):
            for f in sorted(os.listdir(rgb_dir)):
                if f.endswith(".jpg"):
                    base_id = os.path.splitext(f)[0]
                    nir_path = os.path.join(root_dir, "nir", f"{base_id}.png")
                    mask_path = os.path.join(root_dir, "masks", f"{base_id}.png")
                    if os.path.exists(nir_path) and os.path.exists(mask_path):
                        self.sample_names.append(base_id)

    def __len__(self) -> int:
        return len(self.sample_names)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        base_id = self.sample_names[idx]
        rgb_path = os.path.join(self.root_dir, "rgb", f"{base_id}.jpg")
        nir_path = os.path.join(self.root_dir, "nir", f"{base_id}.png")
        mask_path = os.path.join(self.root_dir, "masks", f"{base_id}.png")

        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        nir = cv2.imread(nir_path, cv2.IMREAD_UNCHANGED)
        if len(nir.shape) == 2:
            spectral_band = nir[:, :, np.newaxis]
        else:
            spectral_band = nir[:, :, 0:1]

        image_4ch = np.concatenate([rgb, spectral_band], axis=2)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        if self.is_training and self.augmentation is not None:
            aug_res = self.augmentation(image_4ch, mask)
            image_4ch = aug_res["image"]
            mask = aug_res["mask"]

        image_4ch = cv2.resize(image_4ch, self.img_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        tensor_image = torch.from_numpy(image_4ch).permute(2, 0, 1).float() / 255.0
        tensor_mask = torch.from_numpy(mask).long()

        return {
            "image": tensor_image,
            "mask": tensor_mask,
            "filename": f"{base_id}.png"
        }


class CombinedEWasteDataset(Dataset):
    """
    Unified Multimodal Dataset combining SpectralWaste (Polymers, Packaging, Enclosures)
    and BatteryPackDataset (Class 2 Hazardous Isolation) for end-to-end multi-class coverage.
    """

    def __init__(
        self,
        spectral_dir: str = "data/raw/spectralwaste",
        battery_dir: str = "data/raw/battery_packs",
        img_size: Tuple[int, int] = (640, 640),
        is_training: bool = True,
        augmentation: Optional[IndustrialMultimodalAugmentation] = None
    ):
        self.spectral_ds = SpectralWasteDataset(root_dir=spectral_dir, img_size=img_size, is_training=is_training, augmentation=augmentation)
        self.battery_ds = BatteryPackDataset(root_dir=battery_dir, img_size=img_size, is_training=is_training, augmentation=augmentation)
        self.total_len = len(self.spectral_ds) + len(self.battery_ds)

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < len(self.spectral_ds):
            return self.spectral_ds[idx]
        else:
            return self.battery_ds[idx - len(self.spectral_ds)]


def build_multimodal_dataloader(
    root_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
    img_size: Optional[Tuple[int, int]] = None,
    is_training: bool = True,
    num_workers: int = 0,
    include_batteries: bool = False
) -> DataLoader:
    """
    Constructs an optimized DataLoader for 4-channel multimodal e-waste streams,
    pulling defaults dynamically from system_config.yaml.
    """
    from src.utils.config_loader import CONFIG

    if root_dir is None:
        root_dir = "data/raw/spectralwaste"
    if batch_size is None:
        batch_size = CONFIG.get("training", {}).get("batch_size", 4)
    if img_size is None:
        dim = CONFIG.get("model", {}).get("img_size", 640)
        img_size = (dim, dim)

    conveyor_speed = CONFIG.get("conveyor", {}).get("default_speed_mps", 2.5)
    class_mapping = CONFIG.get("taxonomy", {}).get("dataset_mappings", {}).get("spectralwaste")

    aug = IndustrialMultimodalAugmentation(belt_speed_mps=conveyor_speed) if is_training else None
    
    if include_batteries and os.path.exists("data/raw/battery_packs/rgb"):
        dataset = CombinedEWasteDataset(
            spectral_dir=root_dir,
            battery_dir="data/raw/battery_packs",
            img_size=img_size,
            is_training=is_training,
            augmentation=aug
        )
    else:
        dataset = SpectralWasteDataset(
            root_dir=root_dir,
            img_size=img_size,
            is_training=is_training,
            augmentation=aug,
            class_mapping=class_mapping
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        drop_last=is_training,
        pin_memory=True if torch.cuda.is_available() else False
    )
