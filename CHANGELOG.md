# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project's releases are tracked in the [Release Notes](README.md#-release-notes)
section of the README. This file mirrors the most recent hotfixes and releases.

## [Unreleased]

## [2026.08.30] - Hotfix: Black Output & torch.compile Crash

### Fixed

- **Completely black output with FP8/mixed-precision models** — Removed the
  hardcoded `steps=1 / cfg_scale=1.0` override in `upscale_all_batches`
  (`src/core/generation_phases.py`). Every EMA checkpoint (e.g.
  `seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors`) was being forced
  into a single denoising step with classifier-free guidance disabled. The
  bundled EMA checkpoints are trained for 50 steps at CFG 7.5
  (`configs_7b/main.yaml`, `configs_3b/main.yaml`); a single t=1.0 step with
  no CFG produced a barely-denoised latent that decoded to black. The pipeline
  now uses the configured defaults (50 steps, euler, CFG 7.5) and honors the
  node's `steps` / `sampler_name` widgets.

- **`CompatibleDiT does not support len()` crash with torch.compile** — The
  meta-device checks called `next(runner.dit.parameters())` and truthiness
  tests on the wrapped model. When torch.compile wraps the model in an
  `OptimizedModule`, those calls trigger torch._dynamo's `__len__` guard and
  raise `TypeError`. Added `_unwrap_model()` / `_model_device_type()` helpers
  in `src/core/model_loader.py` that safely unwrap `_orig_mod` /
  `CompatibleDiT` wrappers, and applied them at every meta-device check site:
  VAE encode, DiT upscale, VAE decode, and `materialize_model`.

### Added

- **Diagnostic probes for black-output debugging** — `_log_tensor_stats()`
  in `src/core/generation_phases.py` logs shape/dtype/min/max/mean plus
  NaN/Inf presence at five checkpoints: VAE-encoded latent, DiT output
  latent, VAE-decoded pixels, and Phase 4 pre/post-normalization. Probes
  always print regardless of `enable_debug`; NaN/Inf are logged at ERROR
  level to pinpoint the failing stage.

## [2026.08.26] - Version 2.5.26

- ComfyUI Extension & Tool Compatibility — exported `NODE_CLASS_MAPPINGS`
  and `NODE_DISPLAY_NAME_MAPPINGS` for older ComfyUI/Manager compatibility.
- CompatibleDiT encapsulation fix — robust `__getattr__`/`__setattr__`
  delegation preventing recursion bugs.
- Triton kernel safety & automatic PyTorch fallbacks for CPU/MPS.
- Fused AdaLN text modulation fix (`txt_res` unpack crash).
- RGBA alpha upscaling fix (`guided_filter_pytorch` `NameError`).
- Schema & parameter defaults sync between `INPUT_TYPES` and `execute()`.
- Modernized autocast calls and non-CUDA CUDA-graph guards.
- GGUF dequantization stability via `as_subclass()`.
- Safe BF16 hardware probing for legacy GPUs.

*Earlier releases are listed in the [README Release Notes](README.md#-release-notes).*