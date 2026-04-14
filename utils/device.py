import os

import torch


def get_device() -> torch.device:
    """Detect and return the best available compute device.

    Priority: CUDA (NVIDIA GPU) > MPS (Apple Metal) > CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"VRAM: {vram:.1f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        device = torch.device("mps")
        print("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU detected, using CPU (very slow)")
    return device
