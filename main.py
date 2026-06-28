from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.utils import set_seed, get_device, get_logger, load_config, append_metrics

MODEL_KEYS = ["yolov8", "yolov5", "faster_rcnn", "ssd", "efficientdet", "detr"]
YOLO_KEYS = {"yolov8", "yolov5"}

def run_yolo(key, cfg, logger):
    from src.models.yolov8_model import YOLOv8Model
    from src.models.yolov5_model import YOLOv5Model
    seed = cfg["project"]["seed"]
    model = (YOLOv8Model if key == "yolov8" else YOLOv5Model).from_config(cfg, seed=seed)
    data_yaml = cfg["data"]["yaml"]
    logger.info(f"=== Обучение {key} (Ultralytics) ===")
    model.train(data_yaml, epochs=cfg["train"]["epochs"],
                patience=cfg["train"]["early_stop_patience"])
    metrics = model.evaluate(data_yaml, split="test",
                             class_names=cfg["data"]["class_names"])
    return metrics

def run_pytorch(key, cfg, device, logger):
    from src.dataset.dataset import get_dataloaders
    from src.training.train import train_pytorch_model
    from src.models.faster_rcnn import FasterRCNNModel
    from src.models.ssd import SSDModel
    from src.models.efficientdet import EfficientDetModel
    from src.models.detr import DETRModel

    builders = {"faster_rcnn": FasterRCNNModel, "ssd": SSDModel,
                "efficientdet": EfficientDetModel, "detr": DETRModel}
    model = builders[key].from_config(cfg)

    normalize = cfg["normalize"] if model.needs_normalize else None
    loaders = get_dataloaders(
        cfg["data"]["processed_dir"], img_size=model.img_size,
        batch=cfg["models"][key]["batch"], num_workers=cfg["train"]["num_workers"],
        augment=True, aug_cfg=cfg["augment"], normalize=normalize,
    )
    logger.info(f"=== Обучение {key} (PyTorch цикл) ===")
    metrics, _ = train_pytorch_model(
        model, loaders, device, epochs=cfg["train"]["epochs"],
        patience=cfg["train"]["early_stop_patience"],
        num_classes=cfg["data"]["num_classes"],
        class_names=cfg["data"]["class_names"],
        iou_thr=cfg["eval"]["iou_thr"], score_thr=cfg["eval"]["score_thr"],
        logger=logger,
    )
    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8",
                    choices=MODEL_KEYS + ["all"])
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    device = get_device(cfg["train"]["device"])
    logger = get_logger("train")
    logger.info(f"Устройство: {device}")

    keys = MODEL_KEYS if args.model == "all" else [args.model]
    for key in keys:
        try:
            if key in YOLO_KEYS:
                metrics = run_yolo(key, cfg, logger)
            else:
                metrics = run_pytorch(key, cfg, device, logger)

            row = {"model": key, **{k: v for k, v in metrics.items()
                                    if k != "per_class"}}
            append_metrics(cfg["eval"]["results_csv"], row)
            logger.info(f"{key} → {json.dumps({k: metrics[k] for k in metrics if k!='per_class'}, ensure_ascii=False)}")

            Path("results/logs").mkdir(parents=True, exist_ok=True)
            with open(f"results/logs/{key}_per_class.json", "w", encoding="utf-8") as f:
                json.dump(metrics.get("per_class", {}), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка при обучении {key}: {e}")
            import traceback; logger.error(traceback.format_exc())

    logger.info(f"Готово. Сводка метрик: {cfg['eval']['results_csv']}")

if __name__ == "__main__":
    main()
