import os
import torch


def get_best_device(min_free_gb: float = 2.0) -> torch.device:
    """Return a CUDA device with sufficient free memory, or CPU if unavailable.

    If CUDA_VISIBLE_DEVICES is set to a single GPU (as done by run_all.sh),
    return cuda:0 to respect that assignment directly.
    Otherwise, pick the GPU with the most free memory among those with at
    least min_free_gb free (falls back to best-available if none qualify).
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")

    # If the caller already pinned us to one GPU via CUDA_VISIBLE_DEVICES, use it
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible and len(visible.split(",")) == 1:
        return torch.device("cuda:0")

    n = torch.cuda.device_count()
    min_free_bytes = min_free_gb * (1024 ** 3)
    candidates = [
        (i, torch.cuda.mem_get_info(i)[0])
        for i in range(n)
        if torch.cuda.mem_get_info(i)[0] >= min_free_bytes
    ]
    if not candidates:
        candidates = [(i, torch.cuda.mem_get_info(i)[0]) for i in range(n)]
    best_idx = max(candidates, key=lambda x: x[1])[0]
    return torch.device(f"cuda:{best_idx}")
