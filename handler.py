import os
import traceback

import runpod

from app.logic import edit_image, log_event, warmup_model


if os.getenv("PRELOAD_MODEL", "0") == "1":
    log_event("preload.start", request_id="preload", include_resources=True)
    warmup_model(request_id="preload")
    log_event("preload.done", request_id="preload", include_resources=True)


def handler(job):
    job_input = job.get("input") or {}
    request_id = job.get("id") or job.get("requestId") or job.get("request_id")

    log_event(
        "handler.received",
        request_id=request_id,
        job_keys=sorted(job.keys()),
        input_keys=sorted(job_input.keys()),
        include_resources=True,
    )

    if job_input.get("self_test"):
        log_event("handler.self_test", request_id=request_id)
        return {
            "ok": True,
            "self_test": True,
            "model_id": "unsloth/Qwen-Image-Edit-2511-GGUF::qwen-image-edit-2511-Q2_K.gguf",
            "runtime": "comfyui-gguf",
        }

    try:
        result = edit_image(job_input, request_id=request_id)
        log_event("handler.success", request_id=request_id, include_resources=True)
        return result
    except Exception as exc:
        log_event(
            "handler.error",
            request_id=request_id,
            error=str(exc),
            traceback_text=traceback.format_exc(),
            include_resources=True,
        )
        response = {"ok": False, "error": str(exc)}
        if os.getenv("DEBUG_ERRORS", "0") == "1":
            response["traceback"] = traceback.format_exc()
        return response


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
