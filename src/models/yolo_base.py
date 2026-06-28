from __future__ import annotations

class UltralyticsYOLO:
    name = "yolo"

    def __init__(self, weights, img_size=640, batch=16, lr0=0.01,
                 optimizer="SGD", seed=42, project="results/runs", run_name=None):
        from ultralytics import YOLO
        self.weights = weights
        self.img_size = img_size
        self.batch = batch
        self.lr0 = lr0
        self.optimizer = optimizer
        self.seed = seed
        self.project = project
        self.run_name = run_name or self.name
        self.model = YOLO(weights)

    def train(self, data_yaml, epochs=50, patience=10, device=0):
        self.model.train(
            data=data_yaml, epochs=epochs, imgsz=self.img_size, batch=self.batch,
            lr0=self.lr0, optimizer=self.optimizer, seed=self.seed,
            patience=patience, project=self.project, name=self.run_name,
            device=device, plots=True, exist_ok=True,
        )
        return self.model

    def evaluate(self, data_yaml, split="test", class_names=None):
        res = self.model.val(data=data_yaml, split=split, imgsz=self.img_size,
                             project=self.project, name=f"{self.run_name}_{split}",
                             exist_ok=True)
        box = res.box
        p, r = float(box.mp), float(box.mr)
        f1 = 2 * p * r / max(p + r, 1e-9)
        out = {
            "mAP@0.5": round(float(box.map50), 4),
            "mAP@0.5:0.95": round(float(box.map), 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "per_class": {},
        }

        if class_names is not None:
            try:
                for i, ap in enumerate(box.ap50):
                    nm = class_names[i] if i < len(class_names) else f"class_{i}"
                    out["per_class"][nm] = {"AP": round(float(ap), 4)}
            except Exception:
                pass
        return out

    def predict(self, image, conf=0.25):
        r = self.model.predict(image, imgsz=self.img_size, conf=conf, verbose=False)[0]
        return {
            "boxes": r.boxes.xyxy.cpu(),
            "scores": r.boxes.conf.cpu(),
            "labels": r.boxes.cls.cpu().long(),
        }
