# Детектирование опухолей головного мозга на МРТ-снимках

Учебная практика. Проект полного цикла компьютерного зрения: сравнение
**шести** современных архитектур детектирования объектов на медицинских изображениях.

**Модели:** YOLOv8n · YOLOv5s · Faster R-CNN ResNet50-FPN v2 · SSD MobileNetV3-Large ·
EfficientDet-D0 · DETR ResNet-50.

**Задача:** детектирование (bounding box + класс) на МРТ-снимках. Классы (4):
*Glioma_Tumor*, *Meningioma_Tumor*, *Pituitary_Tumor*, *No_Tumor*.

---

## Структура проекта

```
medical_cv_project/
├── README.md
├── requirements.txt
├── setup.py
├── main.py                            точка входа: python main.py --model <name|all>
├── configs/default.yaml               гиперпараметры датасета и моделей
├── data/
│   ├── raw/                           распакованный экспорт датасета
│   └── processed/                     train/val/test + data.yaml (создаёт prepare_data.py)
├── scripts/prepare_data.py            подготовка данных (COCO/YOLO -> YOLO bbox) + EDA
├── src/
│   ├── dataset/dataset.py             датасет и DataLoader'ы
│   ├── models/                        yolov8, yolov5, faster_rcnn, ssd, efficientdet, detr
│   ├── training/train.py              цикл обучения PyTorch-моделей
│   ├── evaluation/metrics.py          mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1
│   └── utils/utils.py                 seed, логирование, визуализация
├── notebooks/
│   ├── exploration.ipynb              исследовательский анализ данных (EDA)
│   └── Colab_AUTORUN_brain_tumor.ipynb  запуск обучения в Google Colab
└── results/{plots,logs}/              графики и метрики экспериментов
```

---

## Скачивание датасета (без API)

Используется **Brain_Tumor_Detect** (Roboflow Universe, версия v1: 3443 изображения,
4 класса, сплит 3012/287/144). Скачивается **одним zip-файлом, без API-ключей**:

- Страница датасета: https://universe.roboflow.com/mri-brain-tumor-detection/brain_tumor_detect/dataset/1

1. Откройте страницу датасета (ссылка выше).
2. Нажмите **Download Dataset → выберите формат → Download zip to computer**.
   - Рекомендуется формат **YOLOv8** (Object Detection). Формат **COCO** тоже подойдёт —
     `prepare_data.py` понимает оба и сам конвертирует полигоны/COCO в bbox.
3. Загрузите zip на Google Drive в `MyDrive/datasets/brain-tumor.zip`.
4. Дальше — по ноутбуку (распаковка в `data/raw/`).

> Код dataset-agnostic: `prepare_data.py` поддерживает COCO, YOLO-detection и
> YOLO-segmentation; классы и их число берутся из данных и пишутся в `data.yaml`.
> Если число классов отличается от 4 — поправьте `num_classes` и `class_names`
> в `configs/default.yaml` (скрипт подскажет нужные значения в конце вывода).

---

## Запуск (Google Colab, GPU T4)

Готовый ноутбук `notebooks/Colab_AUTORUN_brain_tumor.ipynb` создаёт код проекта,
скачивает датасет и обучает все модели. Откройте его в Colab, выберите среду
**GPU (T4)** и выполните ячейки сверху вниз.

Бесплатный Colab ограничивает длительность сессии (~5 ч). Тяжёлые модели
(Faster R-CNN, DETR) можно обучать в отдельных сессиях; результаты каждой модели
дописываются в `results/logs/metrics.csv`.

### Локальный запуск

```bash
pip install -r requirements.txt
python scripts/prepare_data.py --src data/raw/brain_tumor_detect-1 --dst data/processed
python main.py --model all          # или конкретную модель: --model yolov8
```

---

## Результаты

Сводные метрики всех моделей — в `results/logs/metrics.csv`, поклассовые —
в `results/logs/*_per_class.json`, графики обучения — в `results/plots/`.
Итоговый сравнительный анализ приведён в отчёте `Отчёт_по_практике_Стадник.docx`.

---

## Воспроизводимость

- Глобальный `seed = 42` (`src/utils/utils.py: set_seed`), детерминированный cuDNN.
- Все гиперпараметры зафиксированы в `configs/default.yaml`.
- Разбиение train/val/test фиксировано (детерминированный сплит по seed).

## Лицензия данных

Датасет распространяется на условиях лицензии, указанной на его странице Roboflow - CC BY 4.0.
