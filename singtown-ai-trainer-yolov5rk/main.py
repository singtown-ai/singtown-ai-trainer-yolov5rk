from singtown_ai import SingTownAIClient
from singtown_ai import stdout_watcher, file_watcher
from singtown_ai import export_yolo

client = SingTownAIClient()

@stdout_watcher(interval=1)
def on_stdout_write(content: str):
    client.log(content, end="")

from pathlib import Path
import shutil
import torch
import subprocess

DATASET_PATH = Path("../dataset")
RUNS_PATH = Path("../yolov5")/'runs'
EXP_PATH = RUNS_PATH/'train'/'exp'
METRICS_PATH = EXP_PATH / "results.csv"
shutil.rmtree(RUNS_PATH, ignore_errors=True)

@file_watcher(METRICS_PATH, interval=3)
def file_on_change(content: str):
    import csv
    from io import StringIO

    metrics = list(csv.DictReader(StringIO(content)))
    if not metrics:
        return
    client.update_metrics(metrics)

LABELS = client.task.project.labels
MODEL_NAME = client.task.model_name
EPOCHS = client.task.epochs
BATCH_SIZE = client.task.batch_size
LEARNING_RATE = client.task.learning_rate
EXPORT_WIDTH = client.task.export_width
EXPORT_HEIGHT = client.task.export_height

MODEL_CLASS, IMG_SZ = MODEL_NAME.split("_")

print(f"CUDA available:  {torch.cuda.is_available()}")

print("Download dataset")
export_yolo(client, DATASET_PATH)

print("Training started")
subprocess.run(f"python train.py --data ../dataset/data.yaml --weights ../weights/yolov5s640_coco2017.pt --epochs {EPOCHS} --img {IMG_SZ} --batch-size {BATCH_SIZE}", shell=True, check=True, cwd=Path(__file__).parent.parent/"yolov5")

print("Export onnx")
subprocess.run(f"python export.py --rknpu --weights runs/train/exp/weights/best.pt --img {EXPORT_WIDTH} {EXPORT_HEIGHT}", shell=True, check=True, cwd=Path(__file__).parent.parent/"yolov5")
