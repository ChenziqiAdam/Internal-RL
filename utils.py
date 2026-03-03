import torch


def get_best_device() -> torch.device:
    """Return the CUDA device with the most free memory, or CPU if unavailable."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best_idx = max(
        range(torch.cuda.device_count()),
        key=lambda i: torch.cuda.mem_get_info(i)[0],  # free bytes
    )
    return torch.device(f"cuda:{best_idx}")
