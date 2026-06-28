from __future__ import annotations

from .yolo_base import UltralyticsYOLO

class YOLOv5Model(UltralyticsYOLO):
    name = "yolov5"

    @classmethod
    def from_config(cls, cfg, seed=42):
        m = cfg["models"]["yolov5"]
        return cls(weights=m["weights"], img_size=m["img_size"], batch=m["batch"],
                   lr0=m["lr0"], optimizer=m["optimizer"], seed=seed, run_name="yolov5")
