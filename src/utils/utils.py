from __future__ import annotations

import os
import random
import logging
from pathlib import Path

import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(prefer: str = "cuda") -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_logger(name: str, log_dir: str = "results/logs") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(Path(log_dir) / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

def load_config(path: str = "configs/default.yaml") -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def append_metrics(csv_path: str, row: dict) -> None:
    import pandas as pd
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])
    if Path(csv_path).exists():
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, index=False)

def draw_predictions(image, boxes, labels, scores, class_names, out_path, score_thr=0.3):
    import cv2
    if isinstance(image, (str, Path)):
        image = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
    img = image.copy()
    palette = [(255, 56, 56), (56, 255, 56), (56, 56, 255),
               (255, 200, 0), (200, 0, 255), (0, 200, 200)]
    for box, lab, sc in zip(boxes, labels, scores):
        if sc < score_thr:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        color = palette[int(lab) % len(palette)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        name = class_names[int(lab)] if int(lab) < len(class_names) else str(int(lab))
        cv2.putText(img, f"{name} {sc:.2f}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return img
