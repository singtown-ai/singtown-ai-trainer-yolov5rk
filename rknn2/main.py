from singtown_ai import SingTownAIClient, MOCK_TRAIN_OBJECT_DETECTION, stdout_watcher

# mock_data = MOCK_TRAIN_OBJECT_DETECTION
# mock_data['task'] = {
#     "project": {
#         "labels": ["cat", "dog"],
#         "type": "OBJECT_DETECTION",
#     },
#     "device": "singtown-ai-vision-module",
#     "model_name": "yolov5s_640",
#     "freeze_backbone": True,
#     "batch_size": 16,
#     "epochs": 10,
#     "learning_rate": 0.001,
#     "early_stopping": 3,
#     "export_width": 640,
#     "export_height": 480,
# }
client = SingTownAIClient(
    # mock_data=mock_data,
)

@stdout_watcher(interval=1)
def on_stdout_write(content: str):
    client.log(content, end="")


from pathlib import Path
from rknn.api import RKNN
import os
import zipfile
import torch
import random

RUN_PATH = Path("run")
IMAGES_PATH = Path("../dataset")/"images"/"TRAIN"
RUNS_PATH = Path("../yolov5")/'runs'
EXP_PATH = RUNS_PATH/'train'/'exp'
RUN_PATH.mkdir(parents=True, exist_ok=True)

LABELS = client.task.project.labels
EXPORT_WIDTH = client.task.export_width
EXPORT_HEIGHT = client.task.export_height

print(f"CUDA available:  {torch.cuda.is_available()}")

rknn = RKNN()

rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rv1106")
ret = rknn.load_onnx(str(EXP_PATH / "weights/best.onnx"))
if ret != 0:
    raise Exception("Load model failed!")

with open(RUN_PATH/"dataset.txt", "w") as f:
    imgs = os.listdir(IMAGES_PATH)
    selected_imgs = random.sample(imgs, min(100, len(imgs)))
    for img in selected_imgs:
        f.write(str((IMAGES_PATH/img).absolute()) + "\n")

ret = rknn.build(do_quantization=True, dataset=RUN_PATH/"dataset.txt")
if ret != 0:
    raise Exception("Build model failed!")

ret = rknn.export_rknn(str(RUN_PATH/"best.rknn"))
if ret != 0:
    raise Exception("Export model failed!")

rknn.release()

with open(RUN_PATH/"labels.txt", "wb") as f:
    f.write("\n".join(LABELS).encode("utf-8"))

with zipfile.ZipFile(RUN_PATH/"result.zip", "w") as zipf:
    zipf.write(RUN_PATH/"labels.txt", arcname="labels.txt")
    zipf.write(RUN_PATH/"best.rknn", arcname="best.rknn")
    zipf.write("../yolov5/RK_anchors.txt", arcname="RK_anchors.txt")

client.upload_results_zip(RUN_PATH/"result.zip")
print("Finished")
