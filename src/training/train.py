from __future__ import annotations

import time
from pathlib import Path

import torch
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics

@torch.no_grad()
def evaluate_model(model_wrapper, loader, num_classes, iou_thr=0.5,
                   score_thr=0.05, class_names=None):
    model_wrapper.eval_mode()
    all_preds, all_targets = [], []
    for images, targets in tqdm(loader, desc="eval", leave=False):
        preds = model_wrapper.predict(images)
        for p, t in zip(preds, targets):
            keep = p["scores"] >= score_thr
            all_preds.append({"boxes": p["boxes"][keep],
                              "scores": p["scores"][keep],
                              "labels": p["labels"][keep]})
            all_targets.append({"boxes": t["boxes"], "labels": t["labels"]})
    return compute_metrics(all_preds, all_targets, num_classes,
                           iou_thr=iou_thr, class_names=class_names)

def plot_curves(history, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epochs, history["train_loss"], "-o", ms=3)
    ax[0].set_title("Train loss"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss")
    ax[0].grid(alpha=0.3)
    ax[1].plot(epochs, history["val_map50"], "-o", ms=3, color="green")
    ax[1].set_title("Val mAP@0.5"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("mAP@0.5")
    ax[1].grid(alpha=0.3)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)

def train_pytorch_model(model_wrapper, loaders, device, epochs=50,
                        patience=10, num_classes=3, class_names=None,
                        iou_thr=0.5, score_thr=0.05, logger=None,
                        plots_dir="results/plots"):
    model_wrapper.to(device)
    optimizer, scheduler = model_wrapper.build_optimizer()

    best_map, best_state, no_improve = -1.0, None, 0
    history = {"train_loss": [], "val_map50": []}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model_wrapper.train_mode()
        running, n = 0.0, 0
        for images, targets in tqdm(loaders["train"], desc=f"epoch {epoch}/{epochs}",
                                    leave=False):
            optimizer.zero_grad()
            loss = model_wrapper.train_forward(images, targets)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), 10.0)
            optimizer.step()
            running += float(loss.item()); n += 1
        scheduler.step()
        train_loss = running / max(n, 1)

        val = evaluate_model(model_wrapper, loaders["val"], num_classes,
                             iou_thr, score_thr, class_names)
        history["train_loss"].append(train_loss)
        history["val_map50"].append(val["mAP@0.5"])

        msg = (f"[{model_wrapper.name}] epoch {epoch}: loss={train_loss:.4f} "
               f"val mAP@0.5={val['mAP@0.5']:.4f}")
        (logger.info if logger else print)(msg)

        if val["mAP@0.5"] > best_map:
            best_map = val["mAP@0.5"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in _state_dict(model_wrapper).items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                (logger.info if logger else print)(
                    f"Ранняя остановка на эпохе {epoch} (нет улучшения {patience} эпох)")
                break

    if best_state is not None:
        _load_state_dict(model_wrapper, best_state)

    train_time = time.time() - t0
    plot_curves(history, Path(plots_dir) / f"{model_wrapper.name}_curves.png")

    test = evaluate_model(model_wrapper, loaders["test"], num_classes,
                          iou_thr, score_thr, class_names)
    test["train_time_sec"] = round(train_time, 1)
    return test, history

def _state_dict(wrapper):
    if hasattr(wrapper, "model"):
        return wrapper.model.state_dict()
    return wrapper.bench.state_dict()

def _load_state_dict(wrapper, state):
    if hasattr(wrapper, "model"):
        wrapper.model.load_state_dict(state)
    else:
        wrapper.bench.load_state_dict(state)
