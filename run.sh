set -e

(
  cd singtown-ai-trainer-yolov5rk
  uv run main.py
)

(
  cd rknn2
  uv run main.py
)