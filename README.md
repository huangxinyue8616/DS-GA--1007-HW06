# Flange Defect Inspector

A YOLO-based starter system for detecting industrial defects on flange images. It includes a command-line interface for training, evaluation, and batch/image/video/camera prediction, plus a Streamlit interface for interactive image inspection.

> **Dataset and model notice:** This repository does not include a trained model or images. Prepare and annotate a representative flange dataset before training. The example class names and model settings are starting points; adapt them to your inspection process and validate the model before production use.

## Features

- Train an Ultralytics YOLO detection model from a dataset in YOLO format.
- Run prediction on a single image, a directory, a video, or a camera.
- Evaluate a trained model on the train, validation, or test split.
- Inspect an uploaded image in a browser and review detected class, confidence, and bounding box.
- Configure defect classes and dataset locations in `configs/flange.yaml`.

## Project layout

```text
.
├── app.py
├── configs/
│   └── flange.yaml
├── flange_inspector/
│   ├── __init__.py
│   ├── cli.py
│   └── detector.py
├── requirements.txt
└── tests/
    └── test_detector.py
```

## Requirements and setup

Python 3.10 or newer is recommended. Install PyTorch using the instructions for your operating system and hardware from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/), then install this project's dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Ultralytics can download the initial `yolov8n.pt` weights when training starts, so the first run needs network access. To use a different supported YOLO model or a local checkpoint, pass it with `--model`.

## Prepare a dataset

Organize images and YOLO-format labels as follows:

```text
datasets/
└── flange/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

Each image must have a label file with the same base name (for example, `part_001.jpg` and `part_001.txt`). Each non-empty label line must use the YOLO detection format:

```text
class_id x_center y_center width height
```

Coordinates are normalized to the image width and height and must be between 0 and 1. Class IDs are zero-based and must match the `names` mapping in `configs/flange.yaml`. Update the class names, IDs, and `path` in that file to match the annotated dataset. The example classes are `crack`, `corrosion`, `scratch`, `deformation`, and `missing_bolt`; they are not a prescribed taxonomy.

## Train

From the repository root:

```bash
python -m flange_inspector.cli train \
  --data configs/flange.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16
```

Training outputs are written under `runs/detect/flange/` by default. The best checkpoint is normally `runs/detect/flange/weights/best.pt`. Choose `--device 0` for the first CUDA GPU or `--device cpu` to force CPU execution. Use `--batch -1` for Ultralytics automatic batch-size selection.

## Predict

Predict on one image:

```bash
python -m flange_inspector.cli predict \
  --weights runs/detect/flange/weights/best.pt \
  --source path/to/flange.jpg \
  --conf 0.25
```

The same command accepts a directory or video path. Use `--source 0` for the default camera:

```bash
python -m flange_inspector.cli predict \
  --weights runs/detect/flange/weights/best.pt \
  --source 0
```

Annotated results are saved under `runs/predict/flange/`. Add `--save-txt` to also save YOLO-format prediction labels.

## Evaluate

Evaluate on the test split:

```bash
python -m flange_inspector.cli evaluate \
  --weights runs/detect/flange/weights/best.pt \
  --data configs/flange.yaml \
  --split test
```

Use `--split val` to evaluate on the validation split. Ultralytics reports detection metrics such as precision, recall, and mean average precision.

## Web interface

Start the browser-based image inspector:

```bash
streamlit run app.py
```

Enter the trained checkpoint path in the sidebar, upload a JPG, PNG, BMP, or WebP image, adjust the confidence threshold, and select **Detect defects**. The app displays an annotated image and a table of detections.

## Tests

Run the focused unit tests with:

```bash
python -m pytest
```

## Operational guidance

- Use images representative of the target flange types, lighting, camera angles, and defect severity.
- Split data by physical part, production lot, or acquisition session where possible to reduce leakage between train and test sets.
- Review annotation consistency and class balance before training.
- Select confidence thresholds and acceptance criteria using a separate validation set.
- Measure latency and detection quality on the target hardware. This starter is not a certified safety or quality-control system; production deployment requires domain validation and appropriate human review.
