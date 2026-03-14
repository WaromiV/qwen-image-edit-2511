import base64
import copy
import io
import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode

import requests
from PIL import Image

RUNPOD_VOLUME = Path(os.getenv("RUNPOD_VOLUME", "/runpod-volume"))
COMFYUI_DIR = Path(os.getenv("COMFYUI_DIR", "/opt/ComfyUI"))
COMFY_HOST = os.getenv("COMFY_HOST", "127.0.0.1")
COMFY_PORT = int(os.getenv("COMFY_PORT", "8188"))
COMFY_BASE_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"
WORKFLOW_PATH = Path(
    os.getenv("WORKFLOW_PATH", "/app/workflow/qwen_image_edit_2511_gguf_single.json")
)
GGUF_FILENAME = "qwen-image-edit-2511-Q2_K.gguf"
TEXT_ENCODER_FILENAME = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE_FILENAME = "qwen_image_vae.safetensors"
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "20"))
DEFAULT_TRUE_CFG_SCALE = float(os.getenv("DEFAULT_TRUE_CFG_SCALE", "4.0"))
DEFAULT_WIDTH = int(os.getenv("DEFAULT_WIDTH", "1024"))
DEFAULT_HEIGHT = int(os.getenv("DEFAULT_HEIGHT", "1024"))

if RUNPOD_VOLUME.exists():
    MODEL_ROOT = Path(
        os.getenv(
            "MODEL_ROOT",
            str(RUNPOD_VOLUME / "huggingface" / "qwen-image-edit-2511-gguf"),
        )
    )
else:
    MODEL_ROOT = Path(
        os.getenv("MODEL_ROOT", "/opt/model-cache/qwen-image-edit-2511-gguf")
    )

MODEL_SPECS = [
    {
        "filename": GGUF_FILENAME,
        "url": f"https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/resolve/main/{GGUF_FILENAME}",
        "subdir": "unet",
    },
    {
        "filename": TEXT_ENCODER_FILENAME,
        "url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "subdir": "text_encoders",
    },
    {
        "filename": VAE_FILENAME,
        "url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors",
        "subdir": "vae",
    },
]

START_LOCK = threading.Lock()
SERVER_PROCESS = None


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=600) as response:
        response.raise_for_status()
        with temp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temp_path.replace(destination)


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    try:
        target.symlink_to(source)
    except OSError:
        target.write_bytes(source.read_bytes())


def ensure_model_files() -> None:
    for spec in MODEL_SPECS:
        source = MODEL_ROOT / spec["filename"]
        if not source.exists():
            _download_file(spec["url"], source)
        target = COMFYUI_DIR / "models" / spec["subdir"] / spec["filename"]
        _link_or_copy(source, target)


def _log_path() -> Path:
    return Path(os.getenv("COMFY_LOG_PATH", "/tmp/comfyui-qwen.log"))


def _start_command() -> list[str]:
    return ["/app/start.sh"]


def _server_alive() -> bool:
    global SERVER_PROCESS
    return SERVER_PROCESS is not None and SERVER_PROCESS.poll() is None


def _wait_until_ready(timeout: int = 300) -> None:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            response = requests.get(f"{COMFY_BASE_URL}/system_stats", timeout=5)
            if response.ok:
                return
            last_error = f"HTTP {response.status_code}"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(2)

    log_path = _log_path()
    tail = ""
    if log_path.exists():
        tail = log_path.read_text(errors="ignore")[-4000:]
    raise RuntimeError(f"ComfyUI did not become ready: {last_error}\n{tail}")


def warmup_model() -> None:
    global SERVER_PROCESS
    with START_LOCK:
        if _server_alive():
            return

        ensure_model_files()
        log_path = _log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as log_file:
            SERVER_PROCESS = subprocess.Popen(  # noqa: S603
                _start_command(),
                cwd=COMFYUI_DIR,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
        _wait_until_ready()


def _decode_image(value: str) -> bytes:
    if value.startswith("data:"):
        value = value.split(",", 1)[1]
    return base64.b64decode(value)


def _load_image(job_input: dict) -> Image.Image:
    if job_input.get("image_base64"):
        raw = _decode_image(job_input["image_base64"])
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if job_input.get("image_url"):
        response = requests.get(job_input["image_url"], timeout=120)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    if job_input.get("image_path"):
        return Image.open(job_input["image_path"]).convert("RGB")
    if job_input.get("images"):
        first = job_input["images"][0]
        if isinstance(first, str):
            return Image.open(io.BytesIO(_decode_image(first))).convert("RGB")
        if first.get("image_base64"):
            return Image.open(io.BytesIO(_decode_image(first["image_base64"]))).convert(
                "RGB"
            )
        if first.get("image_url"):
            response = requests.get(first["image_url"], timeout=120)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        if first.get("image_path"):
            return Image.open(first["image_path"]).convert("RGB")
    raise ValueError("Provide image_base64, image_url, image_path, or images")


def _write_input_image(image: Image.Image, job_id: str) -> str:
    filename = f"{job_id}.png"
    image.save(COMFYUI_DIR / "input" / filename, format="PNG")
    return filename


def _load_workflow() -> dict:
    return json.loads(WORKFLOW_PATH.read_text())


def _patch_workflow(
    workflow: dict,
    *,
    input_filename: str,
    prompt: str,
    seed: int,
    steps: int,
    true_cfg_scale: float,
    width: int,
    height: int,
    filename_prefix: str,
) -> dict:
    patched = copy.deepcopy(workflow)
    patched["1"]["inputs"]["image"] = input_filename
    patched["3"]["inputs"]["unet_name"] = GGUF_FILENAME
    patched["6"]["inputs"]["clip_name"] = TEXT_ENCODER_FILENAME
    patched["7"]["inputs"]["vae_name"] = VAE_FILENAME
    patched["8"]["inputs"]["prompt"] = prompt
    patched["10"]["inputs"]["prompt"] = " "
    patched["13"]["inputs"]["seed"] = seed
    patched["13"]["inputs"]["steps"] = steps
    patched["13"]["inputs"]["cfg"] = true_cfg_scale
    patched["13"]["inputs"]["sampler_name"] = "euler"
    patched["13"]["inputs"]["scheduler"] = "simple"
    patched["13"]["inputs"]["denoise"] = 1.0
    patched["15"]["inputs"]["filename_prefix"] = filename_prefix
    patched["16"]["inputs"]["width"] = width
    patched["16"]["inputs"]["height"] = height
    return patched


def _submit_prompt(workflow: dict) -> str:
    payload = {"prompt": workflow, "client_id": str(uuid.uuid4())}
    response = requests.post(f"{COMFY_BASE_URL}/prompt", json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data["prompt_id"]


def _poll_history(prompt_id: str, timeout: int = 1800) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = requests.get(f"{COMFY_BASE_URL}/history/{prompt_id}", timeout=30)
        response.raise_for_status()
        data = response.json()
        if data.get(prompt_id):
            return data[prompt_id]
        time.sleep(2)
    raise TimeoutError(
        f"Timed out waiting for ComfyUI history for prompt_id={prompt_id}"
    )


def _extract_image_metadata(history: dict) -> dict:
    outputs = history.get("outputs") or {}
    images = (outputs.get("15") or {}).get("images") or []
    if not images:
        raise RuntimeError(f"ComfyUI produced no images: {history}")
    return images[0]


def _fetch_output_image(image_meta: dict) -> bytes:
    query = urlencode(
        {
            "filename": image_meta["filename"],
            "subfolder": image_meta.get("subfolder", ""),
            "type": image_meta.get("type", "output"),
        }
    )
    response = requests.get(f"{COMFY_BASE_URL}/view?{query}", timeout=120)
    response.raise_for_status()
    return response.content


def edit_image(job_input: dict) -> dict:
    prompt = (job_input.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Missing required field: prompt")

    warmup_model()

    image = _load_image(job_input)
    width = int(job_input.get("width", DEFAULT_WIDTH))
    height = int(job_input.get("height", DEFAULT_HEIGHT))
    steps = int(
        job_input.get("num_inference_steps", job_input.get("steps", DEFAULT_STEPS))
    )
    true_cfg_scale = float(job_input.get("true_cfg_scale", DEFAULT_TRUE_CFG_SCALE))
    seed = int(job_input.get("seed", 0))
    job_id = uuid.uuid4().hex
    input_filename = _write_input_image(image, job_id)
    workflow = _patch_workflow(
        _load_workflow(),
        input_filename=input_filename,
        prompt=prompt,
        seed=seed,
        steps=steps,
        true_cfg_scale=true_cfg_scale,
        width=width,
        height=height,
        filename_prefix=f"QwenImageEditGGUF_{job_id}",
    )
    prompt_id = _submit_prompt(workflow)
    history = _poll_history(prompt_id)
    image_meta = _extract_image_metadata(history)
    output_bytes = _fetch_output_image(image_meta)
    output_image = Image.open(io.BytesIO(output_bytes))

    return {
        "ok": True,
        "model_id": f"unsloth/Qwen-Image-Edit-2511-GGUF::{GGUF_FILENAME}",
        "runtime": "comfyui-gguf",
        "seed": seed,
        "num_inference_steps": steps,
        "true_cfg_scale": true_cfg_scale,
        "width": output_image.width,
        "height": output_image.height,
        "mime_type": "image/png",
        "image_base64": base64.b64encode(output_bytes).decode("utf-8"),
        "prompt_id": prompt_id,
        "output_filename": image_meta["filename"],
    }
