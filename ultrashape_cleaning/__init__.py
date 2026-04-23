"""ultrashape_cleaning -- Data cleaning pipeline for mesh datasets.

Implements the 4-stage cleaning pipeline described in UltraShape 1.0
(arXiv 2512.21185, §2.1):

    1. Watertightening (sparse voxel + watershed-inspired hole closing + SDF)
    2. VLM-based quality filter (Qwen3-VL-8B)
    3. Pose canonicalization (VLM + geometric hybrid)
    4. Geometry filter (interior/exterior ratio + VAE fragmentation)

This is a FAITHFUL REPLICA, not a 1-for-1 reproduction. The paper's 2048^3
closed-source CUDA kernel is approximated by a sparse-tensor GPU pipeline
built on cubvh (signed distance + ray tracing). See docs/ for per-stage
technical notes and concessions.

Top-level entry points:

    clean_mesh_pipeline(mesh_path) -> CleanResult
    batch_clean(paths, ...) -> list[CleanResult]

CLIs:

    python -m ultrashape_cleaning.clean_mesh --input a.glb --output b.ply --full
    python -m ultrashape_cleaning.batch_clean --input-dir a/ --output-dir b/
"""
from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    "CleanResult",
    "PipelineConfig",
    "clean_mesh_pipeline",
    "batch_clean",
]


def __getattr__(name):
    # Lazy import so `import ultrashape_cleaning` works without pulling in
    # torch/cubvh/transformers at module-load time.
    if name in {"CleanResult", "PipelineConfig", "clean_mesh_pipeline"}:
        from .clean_mesh import CleanResult, PipelineConfig, clean_mesh_pipeline
        return {"CleanResult": CleanResult, "PipelineConfig": PipelineConfig,
                "clean_mesh_pipeline": clean_mesh_pipeline}[name]
    if name == "batch_clean":
        from .batch_clean import batch_clean
        return batch_clean
    raise AttributeError(name)
