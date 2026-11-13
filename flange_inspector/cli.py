"""Command-line tools for training and evaluating flange defect models."""

import argparse
from pathlib import Path
from typing import Sequence

from flange_inspector.detector import FlangeDetector, load_yolo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flange-inspector",
        description="Train and run YOLO flange industrial defect detection.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train a model on a YOLO dataset.")
    train.add_argument("--data", default="configs/flange.yaml", help="Dataset YAML.")
    train.add_argument("--model", default="yolov8n.pt", help="Initial weights or model.")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=16)
    train.add_argument("--device", default=None, help="CUDA device, e.g. 0, or 'cpu'.")
    train.add_argument("--project", default="runs/detect")
    train.add_argument("--name", default="flange")
    train.set_defaults(handler=run_train)

    predict = commands.add_parser(
        "predict", help="Detect defects in images, directories, videos, or a camera."
    )
    predict.add_argument("--weights", required=True, help="Trained .pt weights file.")
    predict.add_argument(
        "--source", required=True, help="Image/video path, directory, or camera index."
    )
    predict.add_argument("--conf", type=float, default=0.25)
    predict.add_argument("--imgsz", type=int, default=640)
    predict.add_argument("--device", default=None)
    predict.add_argument("--project", default="runs/predict")
    predict.add_argument("--name", default="flange")
    predict.add_argument("--save-txt", action="store_true")
    predict.set_defaults(handler=run_predict)

    evaluate = commands.add_parser("evaluate", help="Evaluate weights on a dataset split.")
    evaluate.add_argument("--weights", required=True, help="Trained .pt weights file.")
    evaluate.add_argument("--data", default="configs/flange.yaml", help="Dataset YAML.")
    evaluate.add_argument("--split", choices=("train", "val", "test"), default="test")
    evaluate.add_argument("--imgsz", type=int, default=640)
    evaluate.add_argument("--device", default=None)
    evaluate.set_defaults(handler=run_evaluate)

    return parser


def _require_file(value: str, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _validate_training_options(epochs: int, image_size: int, batch_size: int) -> None:
    if epochs <= 0:
        raise ValueError("Epochs must be a positive integer.")
    if image_size <= 0:
        raise ValueError("Image size must be a positive integer.")
    if batch_size == 0 or batch_size < -1:
        raise ValueError("Batch size must be positive or -1 for automatic selection.")


def run_train(args: argparse.Namespace) -> None:
    data = _require_file(args.data, "Dataset configuration")
    _validate_training_options(args.epochs, args.imgsz, args.batch)
    model = load_yolo(args.model)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )


def run_predict(args: argparse.Namespace) -> None:
    _require_file(args.weights, "Model weights")
    source: str | int = args.source
    source_path = Path(args.source).expanduser()
    if source_path.exists():
        source = str(source_path)
    elif args.source.isdigit():
        source = int(args.source)
    else:
        raise FileNotFoundError(
            f"Prediction source not found: {source_path} "
            "(use an integer such as 0 for a camera)."
        )

    detector = FlangeDetector(args.weights)
    detector.predict(
        source=source,
        confidence=args.conf,
        image_size=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
        save_txt=args.save_txt,
    )


def run_evaluate(args: argparse.Namespace) -> None:
    weights = _require_file(args.weights, "Model weights")
    data = _require_file(args.data, "Dataset configuration")
    if args.imgsz <= 0:
        raise ValueError("Image size must be a positive integer.")
    model = load_yolo(str(weights))
    model.val(
        data=str(data),
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
