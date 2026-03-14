import os
import traceback

import runpod

from app.logic import edit_image, warmup_model


if os.getenv("PRELOAD_MODEL", "0") == "1":
    warmup_model()


def handler(job):
    job_input = job.get("input") or {}

    if job_input.get("self_test"):
        return {
            "ok": True,
            "self_test": True,
            "model_id": "unsloth/Qwen-Image-Edit-2511-GGUF::qwen-image-edit-2511-Q2_K.gguf",
            "runtime": "comfyui-gguf",
        }

    try:
        return edit_image(job_input)
    except Exception as exc:
        response = {"ok": False, "error": str(exc)}
        if os.getenv("DEBUG_ERRORS", "0") == "1":
            response["traceback"] = traceback.format_exc()
        return response


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
