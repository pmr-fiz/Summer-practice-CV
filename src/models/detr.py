from __future__ import annotations

import torch

class DETRModel:
    name = "detr"
    needs_normalize = True

    def __init__(self, num_classes, weights="facebook/detr-resnet-50",
                 img_size=800, lr=1e-4, optimizer="AdamW"):
        from transformers import DetrForObjectDetection
        self.num_classes = num_classes
        self.img_size = img_size
        self.lr = lr
        self.optimizer_name = optimizer
        self.model = DetrForObjectDetection.from_pretrained(
            weights, num_labels=num_classes, ignore_mismatched_sizes=True)
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    def parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def build_optimizer(self):

        backbone = [p for n, p in self.model.named_parameters()
                    if "backbone" in n and p.requires_grad]
        rest = [p for n, p in self.model.named_parameters()
                if "backbone" not in n and p.requires_grad]
        opt = torch.optim.AdamW(
            [{"params": rest, "lr": self.lr},
             {"params": backbone, "lr": self.lr * 0.1}], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
        return opt, sched

    def _build_labels(self, targets):
        labels = []
        for t in targets:
            boxes = t["boxes"].to(self.device)
            if boxes.shape[0] == 0:
                labels.append({
                    "class_labels": torch.zeros(0, dtype=torch.long, device=self.device),
                    "boxes": torch.zeros(0, 4, device=self.device),
                })
                continue

            cx = (boxes[:, 0] + boxes[:, 2]) / 2 / self.img_size
            cy = (boxes[:, 1] + boxes[:, 3]) / 2 / self.img_size
            w = (boxes[:, 2] - boxes[:, 0]) / self.img_size
            h = (boxes[:, 3] - boxes[:, 1]) / self.img_size
            labels.append({
                "class_labels": t["labels"].to(self.device),
                "boxes": torch.stack([cx, cy, w, h], dim=1),
            })
        return labels

    def train_forward(self, images, targets):
        pixel_values = torch.stack([img.to(self.device) for img in images])
        labels = self._build_labels(targets)
        out = self.model(pixel_values=pixel_values, labels=labels)
        return out.loss

    @torch.no_grad()
    def predict(self, images):
        self.model.eval()
        pixel_values = torch.stack([img.to(self.device) for img in images])
        out = self.model(pixel_values=pixel_values)
        logits = out.logits.softmax(-1)[..., :-1]
        boxes = out.pred_boxes
        preds = []
        for i in range(pixel_values.shape[0]):
            scores, labels = logits[i].max(-1)
            b = boxes[i]
            cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
            xyxy = torch.stack([
                (cx - w / 2) * self.img_size, (cy - h / 2) * self.img_size,
                (cx + w / 2) * self.img_size, (cy + h / 2) * self.img_size,
            ], dim=1)
            preds.append({"boxes": xyxy.cpu(), "scores": scores.cpu(),
                          "labels": labels.cpu()})
        return preds

    @classmethod
    def from_config(cls, cfg):
        m = cfg["models"]["detr"]
        return cls(num_classes=cfg["data"]["num_classes"], weights=m["weights"],
                   img_size=m["img_size"], lr=m["lr"], optimizer=m["optimizer"])
