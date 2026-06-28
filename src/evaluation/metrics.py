from __future__ import annotations

import numpy as np
import torch

def box_iou(boxes1, boxes2):
    try:
        from torchvision.ops import box_iou as tv_iou
        return tv_iou(boxes1, boxes2)
    except Exception:
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        union = area1[:, None] + area2[None, :] - inter
        return inter / union.clamp(min=1e-9)

def _ap_from_pr(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

def _match_class(preds, gts, iou_thr):
    n_gt = sum(g.shape[0] for g in gts.values())
    if len(preds) == 0:
        return np.array([]), np.array([]), np.array([]), n_gt

    preds = sorted(preds, key=lambda x: x[0], reverse=True)
    matched = {idx: np.zeros(g.shape[0], dtype=bool) for idx, g in gts.items()}
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    scores = np.array([p[0] for p in preds])

    for i, (score, img_idx, box) in enumerate(preds):
        g = gts.get(img_idx)
        if g is None or g.shape[0] == 0:
            fp[i] = 1
            continue
        ious = box_iou(box.unsqueeze(0), g).squeeze(0)
        best = torch.argmax(ious).item()
        if ious[best].item() >= iou_thr and not matched[img_idx][best]:
            tp[i] = 1
            matched[img_idx][best] = True
        else:
            fp[i] = 1
    return tp, fp, scores, n_gt

def _prf(tp, fp, n_gt):
    p = tp / max(tp + fp, 1e-9)
    r = tp / max(n_gt, 1e-9)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return p, r, f1

def compute_metrics(predictions, targets, num_classes, iou_thr=0.5,
                    iou_range=None, class_names=None):
    if iou_range is None:
        iou_range = [round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]

    per_cls_preds = {c: [] for c in range(num_classes)}
    per_cls_gts = {c: {} for c in range(num_classes)}

    for img_idx, (pred, tgt) in enumerate(zip(predictions, targets)):
        for c in range(num_classes):
            m = tgt["labels"] == c
            if m.sum() > 0:
                per_cls_gts[c][img_idx] = tgt["boxes"][m]
        for box, score, lab in zip(pred["boxes"], pred["scores"], pred["labels"]):
            per_cls_preds[int(lab)].append((float(score), img_idx, box))

    ap_main, ap_coco = [], []

    cls_arrays = {}
    names = {}

    for c in range(num_classes):
        name = class_names[c] if class_names and c < len(class_names) else f"class_{c}"
        names[c] = name
        tp, fp, scores, n_gt = _match_class(per_cls_preds[c], per_cls_gts[c], iou_thr)
        cls_arrays[c] = (scores, tp, fp, n_gt)

        if len(tp) == 0:
            if n_gt > 0:
                ap_main.append(0.0)
        else:
            order = np.argsort(-scores)
            tp_c, fp_c = np.cumsum(tp[order]), np.cumsum(fp[order])
            recall = tp_c / max(n_gt, 1e-9)
            precision = tp_c / np.maximum(tp_c + fp_c, 1e-9)
            ap_main.append(_ap_from_pr(recall, precision))

        if n_gt > 0:
            aps_range = []
            for thr in iou_range:
                tp_t, fp_t, sc_t, ng = _match_class(per_cls_preds[c], per_cls_gts[c], thr)
                if len(tp_t) == 0:
                    aps_range.append(0.0); continue
                o = np.argsort(-sc_t)
                tpc, fpc = np.cumsum(tp_t[o]), np.cumsum(fp_t[o])
                rec = tpc / max(ng, 1e-9)
                prec = tpc / np.maximum(tpc + fpc, 1e-9)
                aps_range.append(_ap_from_pr(rec, prec))
            ap_coco.append(float(np.mean(aps_range)))

    all_scores = np.concatenate([a[0] for a in cls_arrays.values()]) \
        if any(len(a[0]) for a in cls_arrays.values()) else np.array([])
    total_gt = sum(a[3] for a in cls_arrays.values())

    best = {"f1": -1.0, "p": 0.0, "r": 0.0, "conf": 0.25}
    if all_scores.size > 0:

        cand = np.unique(np.round(all_scores, 3))
        if cand.size > 200:
            cand = np.quantile(all_scores, np.linspace(0.0, 0.99, 200))
        for t in cand:
            tp_sum = fp_sum = 0
            for scores, tp, fp, _ in cls_arrays.values():
                if len(scores) == 0:
                    continue
                m = scores >= t
                tp_sum += int(tp[m].sum()); fp_sum += int(fp[m].sum())
            p, r, f1 = _prf(tp_sum, fp_sum, total_gt)
            if f1 > best["f1"]:
                best = {"f1": f1, "p": p, "r": r, "conf": float(t)}

    per_class = {}
    bt = best["conf"]
    for c in range(num_classes):
        scores, tp, fp, n_gt = cls_arrays[c]
        name = names[c]

        if len(tp):
            order = np.argsort(-scores)
            tpc, fpc = np.cumsum(tp[order]), np.cumsum(fp[order])
            rec = tpc / max(n_gt, 1e-9); prec = tpc / np.maximum(tpc + fpc, 1e-9)
            ap = _ap_from_pr(rec, prec)
        else:
            ap = 0.0
        if len(scores):
            m = scores >= bt
            p, r, f1 = _prf(int(tp[m].sum()), int(fp[m].sum()), n_gt)
        else:
            p = r = f1 = 0.0
        per_class[name] = dict(AP=round(ap, 4), precision=round(p, 4),
                               recall=round(r, 4), f1=round(f1, 4), n_gt=n_gt)

    return {
        "mAP@0.5": round(float(np.mean(ap_main)) if ap_main else 0.0, 4),
        "mAP@0.5:0.95": round(float(np.mean(ap_coco)) if ap_coco else 0.0, 4),
        "precision": round(best["p"], 4),
        "recall": round(best["r"], 4),
        "f1": round(best["f1"], 4),
        "conf_thr": round(best["conf"], 3),
        "per_class": per_class,
    }
