import os
import traceback
import uuid

import runpod

from app.logic import edit_image, log_event, warmup_model


if os.getenv("PRELOAD_MODEL", "0") == "1":
    log_event("preload.start", request_id="preload", include_resources=True)
    warmup_model(request_id="preload")
    log_event("preload.done", request_id="preload", include_resources=True)


def handler(job):
    job_input = job.get("input") or {}
    request_id = (
        job.get("id")
        or job.get("requestId")
        or job.get("request_id")
        or uuid.uuid4().hex
    )

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

    batch_entries = job_input.get("batch")
    try:
        if batch_entries is not None:
            if not isinstance(batch_entries, list):
                raise ValueError("batch must be a list of request objects")
            base_input = {k: v for k, v in job_input.items() if k != "batch"}
            batch_results = []
            log_event(
                "handler.batch.start",
                request_id=request_id,
                batch_size=len(batch_entries),
                include_resources=True,
            )
            for index, raw_entry in enumerate(batch_entries):
                if not isinstance(raw_entry, dict):
                    raise ValueError(
                        "each batch entry must be a dict of request fields"
                    )
                entry = {k: v for k, v in raw_entry.items() if k != "batch"}
                sub_input = {**base_input, **entry}
                sub_request_id = f"{request_id}.batch{index}"
                log_event(
                    "handler.batch.item",
                    request_id=request_id,
                    batch_index=index,
                    batch_request_id=sub_request_id,
                )
                result = edit_image(sub_input, request_id=sub_request_id)
                combined = {"request_id": sub_request_id, **result}
                batch_results.append(combined)
            log_event(
                "handler.batch.done",
                request_id=request_id,
                batch_size=len(batch_results),
                include_resources=True,
            )
            return {"ok": True, "batch": batch_results}
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
