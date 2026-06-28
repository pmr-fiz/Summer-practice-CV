from __future__ import annotations

import torch

class FasterRCNNModel:
    name = "faster_rcnn"
    needs_normalize = False

    def __init__(self, num_classes, img_size=640, lr=0.005,
                 step_size=15, gamma=0.1, optimizer="SGD"):
        import torchvision
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        self.num_classes = num_classes
        self.img_size = img_size
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
        self.optimizer_name = optimizer

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights="DEFAULT", min_size=img_size, max_size=img_size)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
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
        if self.optimizer_name.upper() == "SGD":
            opt = torch.optim.SGD(self.parameters(), lr=self.lr,
                                  momentum=0.9, weight_decay=5e-4)
        else:
            opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=self.step_size,
                                                gamma=self.gamma)
        return opt, sched

    def _to_targets(self, targets):
        out = []
        for t in targets:
            out.append({
                "boxes": t["boxes"].to(self.device),
                "labels": (t["labels"] + 1).to(self.device),
            })
        return out

    def train_forward(self, images, targets):
        images = [img.to(self.device) for img in images]

        valid = [(im, t) for im, t in zip(images, targets) if t["boxes"].shape[0] > 0]
        if not valid:
            return torch.zeros(1, requires_grad=True, device=self.device).sum()
        images = [v[0] for v in valid]
        targets = self._to_targets([v[1] for v in valid])
        loss_dict = self.model(images, targets)
        return sum(loss for loss in loss_dict.values())

    @torch.no_grad()
    def predict(self, images):
        self.model.eval()
        images = [img.to(self.device) for img in images]
        outputs = self.model(images)
        preds = []
        for o in outputs:
            preds.append({
                "boxes": o["boxes"].cpu(),
                "scores": o["scores"].cpu(),
                "labels": (o["labels"] - 1).cpu(),
            })
        return preds

    @classmethod
    def from_config(cls, cfg):
        m = cfg["models"]["faster_rcnn"]
        return cls(num_classes=cfg["data"]["num_classes"], img_size=m["img_size"],
                   lr=m["lr"], step_size=m["step_size"], gamma=m["gamma"],
                   optimizer=m["optimizer"])
