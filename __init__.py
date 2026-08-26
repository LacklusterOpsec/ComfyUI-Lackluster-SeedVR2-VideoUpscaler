"""
ComfyUI-SeedVR2_VideoUpscaler
Official SeedVR2 integration for ComfyUI
"""

from .src.optimization.compatibility import ensure_triton_compat  # noqa: F401
from .src.interfaces import (
    comfy_entrypoint,
    SeedVR2Extension,
    SeedVR2VideoUpscaler,
    SeedVR2LoadDiTModel,
    SeedVR2LoadVAEModel,
    SeedVR2TorchCompileSettings,
)

NODE_CLASS_MAPPINGS = {
    "SeedVR2VideoUpscaler": SeedVR2VideoUpscaler,
    "SeedVR2LoadDiTModel": SeedVR2LoadDiTModel,
    "SeedVR2LoadVAEModel": SeedVR2LoadVAEModel,
    "SeedVR2TorchCompileSettings": SeedVR2TorchCompileSettings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2VideoUpscaler": "SeedVR2 Video Upscaler",
    "SeedVR2LoadDiTModel": "SeedVR2 Load DiT Model",
    "SeedVR2LoadVAEModel": "SeedVR2 Load VAE Model",
    "SeedVR2TorchCompileSettings": "SeedVR2 Torch Compile Settings",
}

__all__ = [
    "comfy_entrypoint",
    "SeedVR2Extension",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]