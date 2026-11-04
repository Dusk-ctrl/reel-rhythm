"""GPU check so I do not silently attempt Gemma training on my Mac CPU."""


def gpu_report(require_gpu=False):
    try:
        import torch
    except ImportError:
        if require_gpu:
            raise RuntimeError("Install PyTorch on your GPU machine before training.")
        return {"device": "unavailable", "note": "PyTorch is not installed"}
    if not torch.cuda.is_available():
        if require_gpu:
            raise RuntimeError(
                "CUDA GPU not detected. Gemma 4 training is disabled on this Mac. "
                "Use a configured NVIDIA GPU environment. See GPU_PLAN.md."
            )
        return {"device": "CPU/MPS", "note": "Not selected for full Gemma 4 RL training"}
    info = torch.cuda.get_device_properties(0)
    return {"device": info.name, "memory_gib": round(info.total_memory / 2**30, 1),
            "cuda": torch.version.cuda, "pytorch": torch.__version__}


def require_training_gpu():
    report = gpu_report(require_gpu=True)
    print("Detected training hardware:", report)
    if report["memory_gib"] < 48:
        raise RuntimeError(
            "This starter configuration assumes >=48 GiB CUDA VRAM; 80 GiB is the proposed target. "
            "A 24 GiB QLoRA setup needs separate architecture-specific testing; do not just run this script."
        )
    return report
