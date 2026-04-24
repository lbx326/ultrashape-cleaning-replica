"""_config.py -- Resolve cluster-specific paths via environment variables.

The original implementation hard-coded absolute paths to cluster-specific
locations (e.g. ``/moganshan/afs_a/...``). That prevents external clones
from running. This module centralises every such path behind an
environment variable with a sensible public default.

Environment variables (all optional; unset values fall back to defaults):

    ULTRASHAPE_VAE_CONFIG   Path to ``infer_dit_refine.yaml`` (UltraShape).
    ULTRASHAPE_VAE_CKPT     Path to ``ultrashape_v1.pt``.
    ULTRASHAPE_REPO_ROOT    Path to the UltraShape repo (for ``sys.path``
                            injection in ``filter_geometry.py``). Only
                            needed when the UltraShape package is not
                            already importable.
    QWEN3VL_MODEL_PATH      Directory or HuggingFace ID of the Qwen3-VL
                            model. Defaults to the HF hub id.
    VLM_PYTHON_EXE          Python interpreter for the Stage 2 sidecar.
                            When unset, Stage 2 runs in-process.
    VLM_SIDECAR_CWD         Working directory for the sidecar subprocess.
                            Defaults to the repo root (inferred from this
                            module's location).
    VLM_SIDECAR_CUDA_VISIBLE_DEVICES
                            Propagated to the sidecar as CUDA_VISIBLE_DEVICES.

``resolve_paths()`` returns an immutable snapshot for consumers that want
every value at once. Individual callers should prefer the ``get_*``
helpers so the default can evolve without touching call sites.

The goal is that ``pip install -e .`` then ``python -m
ultrashape_cleaning.clean_mesh --help`` succeeds on a clean clone with
*no* environment variables set. Actual runs still need the weights, but
users should see a readable error pointing them at the env var instead
of a mystifying ``FileNotFoundError: /moganshan/...``.
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Optional


# Reasonable public defaults. None of these point at private paths.
#
# - Qwen3-VL defaults to the HuggingFace hub id so ``from_pretrained``
#   can download on first use if the user has HF_HOME set up.
# - VAE config/ckpt default to ``None``; callers raise a readable error
#   if the user didn't set ULTRASHAPE_VAE_{CONFIG,CKPT}.
# - VLM_PYTHON_EXE defaults to ``None`` meaning "run in-process"; the
#   two-env split is cluster-specific and most users don't need it.
_DEFAULT_QWEN3VL = "Qwen/Qwen3-VL-8B-Instruct"


def _repo_root() -> Path:
    """Absolute path to the repo root (parent of the package directory)."""
    return Path(__file__).resolve().parent.parent


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Look up ``name`` in os.environ, returning ``default`` if unset/blank."""
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        return default
    return val


def get_vae_config_path() -> Optional[str]:
    """Path to UltraShape VAE config YAML, or None if not configured."""
    return _env("ULTRASHAPE_VAE_CONFIG")


def get_vae_ckpt_path() -> Optional[str]:
    """Path to UltraShape VAE .pt checkpoint, or None if not configured."""
    return _env("ULTRASHAPE_VAE_CKPT")


def get_ultrashape_repo_root() -> Optional[str]:
    """Path to the UltraShape repo root, or None if not configured."""
    return _env("ULTRASHAPE_REPO_ROOT")


def get_qwen3vl_model_path() -> str:
    """Path or HuggingFace id for Qwen3-VL-8B-Instruct.

    Always returns a non-None string; defaults to the HF hub id.
    """
    return _env("QWEN3VL_MODEL_PATH", _DEFAULT_QWEN3VL) or _DEFAULT_QWEN3VL


def get_vlm_python_exe() -> Optional[str]:
    """Python interpreter for Stage 2 sidecar, or None for in-process."""
    return _env("VLM_PYTHON_EXE")


def get_vlm_sidecar_cwd() -> str:
    """Working directory for Stage 2 sidecar. Defaults to the repo root."""
    v = _env("VLM_SIDECAR_CWD")
    if v:
        return v
    return str(_repo_root())


def get_vlm_sidecar_cuda_visible_devices() -> Optional[str]:
    """CUDA_VISIBLE_DEVICES for the Stage 2 sidecar, or None to inherit."""
    return _env("VLM_SIDECAR_CUDA_VISIBLE_DEVICES")


@dataclasses.dataclass(frozen=True)
class ResolvedPaths:
    """Immutable snapshot of all resolved config paths."""
    vae_config: Optional[str]
    vae_ckpt: Optional[str]
    ultrashape_repo_root: Optional[str]
    qwen3vl_model_path: str
    vlm_python_exe: Optional[str]
    vlm_sidecar_cwd: str
    vlm_sidecar_cuda_visible_devices: Optional[str]


def resolve_paths() -> ResolvedPaths:
    """Snapshot of every configured path. Called once per CLI entry."""
    return ResolvedPaths(
        vae_config=get_vae_config_path(),
        vae_ckpt=get_vae_ckpt_path(),
        ultrashape_repo_root=get_ultrashape_repo_root(),
        qwen3vl_model_path=get_qwen3vl_model_path(),
        vlm_python_exe=get_vlm_python_exe(),
        vlm_sidecar_cwd=get_vlm_sidecar_cwd(),
        vlm_sidecar_cuda_visible_devices=get_vlm_sidecar_cuda_visible_devices(),
    )


__all__ = [
    "ResolvedPaths",
    "get_qwen3vl_model_path",
    "get_ultrashape_repo_root",
    "get_vae_ckpt_path",
    "get_vae_config_path",
    "get_vlm_python_exe",
    "get_vlm_sidecar_cuda_visible_devices",
    "get_vlm_sidecar_cwd",
    "resolve_paths",
]
