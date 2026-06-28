from __future__ import annotations

import torch

class SSDModel:
    name = "ssd"
    needs_normalize = False

    def __init__(self, num_classes, img_size=320, lr=0.01, optimizer="SGD"):
        import torchvision
        from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
        from torchvision.models.detection import _utils as det_utils
        from functools import partial

        self.num_classes = num_classes
        self.img_size = img_size
        self.lr = lr
        self.optimizer_name = optimizer

        self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
            weights="DEFAULT")

        in_channels = det_utils.retrieve_out_channels(self.model.backbone,
                                                       (img_size, img_size))
        num_anchors = self.model.anchor_generator.num_anchors_per_location()
        norm_layer = partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.03)
        self.model.head.classification_head = SSDLiteClassificationHead(
            in_channels, num_anchors, num_classes + 1, norm_layer)

        self.model.transform.min_size = (img_size,)
        self.model.transform.max_size = img_size
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
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
        return opt, sched

    def _to_targets(self, targets):
        return [{"boxes": t["boxes"].to(self.device),
                 "labels": (t["labels"] + 1).to(self.device)} for t in targets]

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
        return [{"boxes": o["boxes"].cpu(), "scores": o["scores"].cpu(),
                 "labels": (o["labels"] - 1).cpu()} for o in outputs]

    @classmethod
    def from_config(cls, cfg):
        m = cfg["models"]["ssd"]
        return cls(num_classes=cfg["data"]["num_classes"], img_size=m["img_size"],
                   lr=m["lr"], optimizer=m["optimizer"])
