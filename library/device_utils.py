import functools
import gc
import os

import torch

try:
    HAS_CUDA = torch.cuda.is_available()
except Exception:
    HAS_CUDA = False

try:
    HAS_MPS = torch.backends.mps.is_available()
except Exception:
    HAS_MPS = False


def clean_memory():
    gc.collect()
    if HAS_CUDA:
        torch.cuda.empty_cache()
    if HAS_MPS:
        torch.mps.empty_cache()


@functools.lru_cache(maxsize=None)
def get_preferred_device_name(purpose: str = "") -> str:
    if purpose:
        envvar = f"SD_{purpose.upper()}_DEVICE"
        torch_purpose_device = os.environ.get(envvar)
        if torch_purpose_device is not None:
            print(f"get_preferred_device_name({purpose}) -> {torch_purpose_device} (from {envvar})")
            return torch_purpose_device
    torch_device = os.environ.get("SD_DEVICE")
    if torch_device is None:
        if HAS_CUDA:
            torch_device = "cuda"
        elif HAS_MPS:
            torch_device = "mps"
        else:
            torch_device = "cpu"
    print(f"get_preferred_device_name({purpose}) -> {torch_device}")
    return torch_device


def init_ipex():
    try:
        import intel_extension_for_pytorch as ipex  # noqa
    except ImportError:
        return

    try:
        if torch.xpu.is_available():
            from library.ipex import ipex_init

            ipex_init()
    except Exception as e:
        print("failed to initialize ipex:", e)
