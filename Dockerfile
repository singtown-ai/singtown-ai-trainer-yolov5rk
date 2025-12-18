FROM ghcr.io/astral-sh/uv:debian

RUN apt update && apt install -y libgl1 cmake && \
    rm -rf /var/lib/apt/lists/*

ADD https://ultralytics.com/assets/Arial.ttf https://ultralytics.com/assets/Arial.Unicode.ttf /root/.config/Ultralytics/

WORKDIR /app

RUN git clone https://github.com/SingTown/yolov5_rk.git yolov5 && rm -rf yolov5/.git && rm yolov5/requirements.txt

COPY . .

RUN cd singtown-ai-trainer-yolov5rk && uv sync
RUN cd rknn2 && uv sync && uv pip install ./rknn_toolkit2-1.5.2+b642f30c-cp310-cp310-linux_x86_64.whl

CMD ["sh", "run.sh"]
