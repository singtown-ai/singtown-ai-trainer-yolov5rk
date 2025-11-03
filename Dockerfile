FROM ghcr.io/astral-sh/uv:debian

RUN apt update && apt install -y libgl1 cmake && \
    rm -rf /var/lib/apt/lists/*

ADD https://ultralytics.com/assets/Arial.ttf https://ultralytics.com/assets/Arial.Unicode.ttf /root/.config/Ultralytics/

WORKDIR /app
COPY . /app

RUN cd singtown-ai-trainer-yolov5rk && uv sync
RUN cd rknn2 && uv sync

CMD ["sh", "run.sh"]
