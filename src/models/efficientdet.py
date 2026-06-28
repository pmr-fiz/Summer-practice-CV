from __future__ import annotations

import torch

class EfficientDetModel:
    name = "efficientdet"
    needs_normalize = True

    def __init__(self, num_classes, variant="tf_efficientdet_d0", img_size=512,
                 lr=1e-4, optimizer="AdamW"):
        from effdet import create_model
        self.num_classes = num_classes
        self.img_size = img_size
        self.lr = lr
        self.optimizer_name = optimizer

        self.bench = create_model(
            variant, bench_task="train", num_classes=num_classes,
            pretrained=True, image_size=(img_size, img_size),
            bench_labeler=True,
        )
        self._predict_bench = None
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = device
        self.bench.to(device)
        return self

    def train_mode(self):
        self.bench.train()

    def eval_mode(self):
        self.bench.eval()

    def parameters(self):
        return [p for p in self.bench.parameters() if p.requires_grad]

    def build_optimizer(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
        return opt, sched

    def _build_target(self, targets, batch_size):
        max_n = max((t["boxes"].shape[0] for t in targets), default=1)
        max_n = max(max_n, 1)
        bbox = torch.zeros(batch_size, max_n, 4, device=self.device)
        cls = torch.zeros(batch_size, max_n, device=self.device)
        for i, t in enumerate(targets):
            n = t["boxes"].shape[0]
            if n == 0:
                continue
            b = t["boxes"].to(self.device)

            bbox[i, :n] = b[:, [1, 0, 3, 2]]
            cls[i, :n] = (t["labels"].to(self.device) + 1).float()
        return {
            "bbox": bbox,
            "cls": cls,
            "img_size": torch.tensor([[self.img_size, self.img_size]] * batch_size,
                                     device=self.device).float(),
            "img_scale": torch.ones(batch_size, device=self.device),
        }

    def train_forward(self, images, targets):
        x = torch.stack([img.to(self.device) for img in images])
        target = self._build_target(targets, x.shape[0])
        out = self.bench(x, target)
        return out["loss"]

    @torch.no_grad()
    def predict(self, images):
        from effdet import DetBenchPredict
        if self._predict_bench is None:
            self._predict_bench = DetBenchPredict(self.bench.model).to(self.device)
        self._predict_bench.eval()
        x = torch.stack([img.to(self.device) for img in images])
        det = self._predict_bench(x)
        preds = []
        for d in det:
            d = d.cpu()
            preds.append({
                "boxes": d[:, :4],
                "scores": d[:, 4],
                "labels": (d[:, 5].long() - 1),
            })
        return preds

    @classmethod
    def from_config(cls, cfg):
        m = cfg["models"]["efficientdet"]
        return cls(num_classes=cfg["data"]["num_classes"], variant=m["variant"],
                   img_size=m["img_size"], lr=m["lr"], optimizer=m["optimizer"])
