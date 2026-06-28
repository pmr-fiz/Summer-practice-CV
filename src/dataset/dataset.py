from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def _list_images(images_dir: Path):
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS])

def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2

class ObjectDetectionDataset(Dataset):

    def __init__(self, split_dir, img_size=640, augment=False, aug_cfg=None,
                 normalize=None):
        self.split_dir = Path(split_dir)
        self.images_dir = self.split_dir / "images"
        self.labels_dir = self.split_dir / "labels"
        self.img_size = img_size
        self.augment = augment
        self.aug = aug_cfg or {}
        self.normalize = normalize
        self.images = _list_images(self.images_dir)
        if not self.images:
            raise FileNotFoundError(f"Не найдено изображений в {self.images_dir}")

    def __len__(self):
        return len(self.images)

    def _read_label(self, img_path: Path):
        label_path = self.labels_dir / (img_path.stem + ".txt")
        boxes, labels = [], []
        if label_path.exists():
            for line in label_path.read_text().strip().splitlines():
                if not line.strip():
                    continue
                cls, cx, cy, w, h = map(float, line.split()[:5])
                boxes.append([cx, cy, w, h])
                labels.append(int(cls))
        return np.array(boxes, dtype=np.float32).reshape(-1, 4), \
               np.array(labels, dtype=np.int64)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        boxes_n, labels = self._read_label(img_path)

        img = cv2.resize(img, (self.img_size, self.img_size))

        if self.augment and boxes_n.shape[0] > 0:
            if np.random.rand() < self.aug.get("hflip_p", 0.0):
                img = img[:, ::-1, :].copy()
                boxes_n[:, 0] = 1.0 - boxes_n[:, 0]
            bc = self.aug.get("brightness_contrast", 0.0)
            if bc > 0:
                alpha = 1.0 + np.random.uniform(-bc, bc)
                beta = np.random.uniform(-bc, bc) * 255
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        if boxes_n.shape[0] > 0:
            xyxy = np.zeros_like(boxes_n)
            for i, (cx, cy, w, h) in enumerate(boxes_n):
                xyxy[i] = yolo_to_xyxy(cx, cy, w, h, self.img_size, self.img_size)
            xyxy = np.clip(xyxy, 0, self.img_size)
        else:
            xyxy = np.zeros((0, 4), dtype=np.float32)

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if self.normalize:
            mean = torch.tensor(self.normalize["mean"]).view(3, 1, 1)
            std = torch.tensor(self.normalize["std"]).view(3, 1, 1)
            img_t = (img_t - mean) / std

        target = {
            "boxes": torch.as_tensor(xyxy, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "orig_size": torch.tensor([orig_h, orig_w]),
        }
        return img_t, target

def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)

def get_dataloaders(processed_dir, img_size=640, batch=8, num_workers=2,
                    augment=True, aug_cfg=None, normalize=None,
                    splits=("train", "val", "test")):
    processed_dir = Path(processed_dir)
    loaders = {}
    for split in splits:
        ds = ObjectDetectionDataset(
            processed_dir / split, img_size=img_size,
            augment=(augment and split == "train"),
            aug_cfg=aug_cfg, normalize=normalize,
        )
        loaders[split] = DataLoader(
            ds, batch_size=batch, shuffle=(split == "train"),
            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
        )
    return loaders
