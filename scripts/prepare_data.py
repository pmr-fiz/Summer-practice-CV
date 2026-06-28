from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SPLIT_ALIASES = {"train": "train", "training": "train",
                 "valid": "val", "val": "val", "validation": "val",
                 "test": "test", "testing": "test"}

def imgs_in(d: Path):
    return sorted([p for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS])

def detect_splits(src: Path):
    out = {}
    for child in src.iterdir():
        if child.is_dir() and child.name.lower() in SPLIT_ALIASES:
            out[SPLIT_ALIASES[child.name.lower()]] = child
    return out

def poly_to_bbox(coords):
    xs = coords[0::2]
    ys = coords[1::2]
    return min(xs), min(ys), max(xs), max(ys)

def convert_coco(split_dir: Path, out_dir: Path, name2idx: dict):
    js = list(split_dir.glob("*_annotations.coco.json")) + list(split_dir.glob("*.json"))
    data = json.loads(js[0].read_text())
    images = {im["id"]: im for im in data["images"]}
    cats = {c["id"]: c["name"] for c in data["categories"]}
    anns_by_img = {}
    for a in data["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    counter = Counter()

    for img_id, im in images.items():
        w, h = im["width"], im["height"]
        fn = im["file_name"]
        src_img = split_dir / fn
        if not src_img.exists():
            cand = list(split_dir.rglob(Path(fn).name))
            if not cand:
                continue
            src_img = cand[0]
        stem = Path(fn).stem
        shutil.copy2(src_img, out_dir / "images" / src_img.name)
        lines = []
        for a in anns_by_img.get(img_id, []):
            cname = cats.get(a["category_id"], None)
            if cname is None or cname not in name2idx:
                continue
            x, y, bw, bh = a["bbox"]
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            lines.append(f"{name2idx[cname]} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
            counter[cname] += 1
        (out_dir / "labels" / f"{stem}.txt").write_text("\n".join(lines))
    return counter

def convert_yolo(split_dir: Path, out_dir: Path, names: list):
    img_dir = split_dir / "images" if (split_dir / "images").exists() else split_dir
    lab_dir = split_dir / "labels" if (split_dir / "labels").exists() else split_dir
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    counter = Counter()
    for img in imgs_in(img_dir):
        shutil.copy2(img, out_dir / "images" / img.name)
        lab = lab_dir / (img.stem + ".txt")
        lines = []
        if lab.exists():
            for line in lab.read_text().strip().splitlines():
                t = line.split()
                if len(t) < 5:
                    continue
                cls = int(float(t[0]))
                vals = list(map(float, t[1:]))
                if len(vals) == 4:
                    cx, cy, bw, bh = vals
                else:
                    x1, y1, x2, y2 = poly_to_bbox(vals)
                    cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                counter[names[cls] if cls < len(names) else cls] += 1
        (out_dir / "labels" / f"{img.stem}.txt").write_text("\n".join(lines))
    return counter

def collect_class_names(src: Path, splits: dict):

    for d in splits.values():
        js = list(d.glob("*_annotations.coco.json")) + list(d.glob("*.json"))
        if js:
            data = json.loads(js[0].read_text())

            names = [c["name"] for c in data["categories"]
                     if str(c.get("supercategory", "none")).lower() != "none"]
            if not names:
                names = [c["name"] for c in data["categories"]]
            return sorted(set(names))

    y = src / "data.yaml"
    if y.exists():
        names = yaml.safe_load(y.read_text()).get("names")
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        if names:
            return sorted(names)
    return None

def is_coco(split_dir: Path):
    return bool(list(split_dir.glob("*_annotations.coco.json")) or list(split_dir.glob("*.json")))

def random_split(src: Path, dst: Path, names, ratios=(0.7, 0.15, 0.15), seed=42):
    img_dir = src / "images" if (src / "images").exists() else src
    images = imgs_in(img_dir)
    random.Random(seed).shuffle(images)
    n = len(images); n_tr = int(n * ratios[0]); n_val = int(n * ratios[1])
    parts = {"train": images[:n_tr], "val": images[n_tr:n_tr + n_val], "test": images[n_tr + n_val:]}
    lab_dir = src / "labels" if (src / "labels").exists() else src
    cnt = Counter()
    for split, files in parts.items():
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)
        for img in files:
            shutil.copy2(img, dst / split / "images" / img.name)
            lab = lab_dir / (img.stem + ".txt")
            if lab.exists():
                shutil.copy2(lab, dst / split / "labels" / lab.name)
    return cnt

def eda(dst: Path, names):
    print("\n=== Анализ датасета (EDA) ===")
    total_i = total_b = 0
    for split in ("train", "val", "test"):
        img_dir = dst / split / "images"
        if not img_dir.exists():
            continue
        imgs = imgs_in(img_dir)
        cls = Counter(); nb = 0
        for lab in (dst / split / "labels").glob("*.txt"):
            for line in lab.read_text().strip().splitlines():
                if line.strip():
                    c = int(float(line.split()[0]))
                    cls[names[c] if c < len(names) else c] += 1; nb += 1
        total_i += len(imgs); total_b += nb
        print(f"  {split:5s}: {len(imgs):5d} изобр., {nb:5d} объектов | {dict(cls)}")
    print(f"  ИТОГО: {total_i} изображений, {total_b} объектов")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="распакованный экспорт Roboflow")
    ap.add_argument("--dst", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    splits = detect_splits(src)
    names = collect_class_names(src, splits) or \
        ["Glioma_Tumor", "Meningioma_Tumor", "No_Tumor", "Pituitary_Tumor"]
    name2idx = {n: i for i, n in enumerate(names)}
    print(f"Классы ({len(names)}): {names}")

    if splits:
        for canon, d in splits.items():
            if is_coco(d):
                convert_coco(d, dst / canon, name2idx)
            else:
                convert_yolo(d, dst / canon, names)
        if "val" not in splits or "test" not in splits:
            print("ВНИМАНИЕ: отсутствует val или test — проверьте экспорт.")
    else:
        print("Деление на выборки не найдено → случайный сплит 70/15/15")
        random_split(src, dst, names, seed=args.seed)

    data_yaml = {"path": str(dst.resolve()), "train": "train/images",
                 "val": "val/images", "test": "test/images",
                 "nc": len(names), "names": names}
    (dst / "data.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True))
    print(f"\nСоздан {dst / 'data.yaml'}")
    print(f">>> ПРОВЕРЬТЕ: в configs/default.yaml num_classes = {len(names)} и "
          f"class_names = {names}")
    eda(dst, names)

if __name__ == "__main__":
    main()
