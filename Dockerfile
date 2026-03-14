FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    COMFYUI_DIR=/opt/ComfyUI \
    COMFY_HOST=127.0.0.1 \
    COMFY_PORT=8188 \
    HF_HOME=/runpod-volume/huggingface \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface/hub \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface/hub \
    MODEL_ROOT=/runpod-volume/huggingface/qwen-image-edit-2511-gguf \
    PRELOAD_MODEL=0

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision && \
    pip install -r /app/requirements.txt && \
    git clone https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI && \
    git clone https://github.com/city96/ComfyUI-GGUF /opt/ComfyUI/custom_nodes/ComfyUI-GGUF && \
    pip install -r /opt/ComfyUI/requirements.txt && \
    pip install -r /opt/ComfyUI/custom_nodes/ComfyUI-GGUF/requirements.txt

COPY handler.py /app/handler.py
COPY app /app/app
COPY workflow /app/workflow
COPY README.md /app/README.md
COPY test_input.json /app/test_input.json
COPY start.sh /app/start.sh

RUN chmod +x /app/start.sh

CMD ["python3", "-u", "handler.py"]
