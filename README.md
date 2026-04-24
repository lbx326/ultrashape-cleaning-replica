# ultrashape-cleaning-replica

A faithful open-source replica of the **UltraShape 1.0** mesh data-cleaning
pipeline described in §2.1 of the paper (arXiv `2512.21185`), built so the
code can actually be run on HSSD-scale datasets without the paper's
closed-source 2048³ CUDA kernel.

> **Paper**: "UltraShape: Ultra-high-resolution 3D Shape Generation with
> Adaptive Latent Refinement" (2025). The authors release only the
> *generation* step (`scripts/sampling.py`); the 4-stage data cleaning
> pipeline they describe for preparing the training corpus is closed.
> This repository reproduces that pipeline as faithfully as possible.

## The four stages

| # | Stage | Paper | This repo |
|---|-------|-------|-----------|
| 1 | Watertightening | CUDA-parallel sparse voxel 2048³ + watershed-like hole closing + volumetric thickening + SDF + MC | Dense GPU voxelization at **1024³** via `cubvh`, max-pool-based watershed flood-fill + closing, adaptive thickening for open shells, `cubvh.signed_distance` + flood-fill sign-correction, `skimage.marching_cubes`. Handles 1M-face inputs at ~90 s on A100-80GB. |
| 2 | VLM quality filter | Multi-view render + VLM | **Qwen3-VL-8B-Instruct** (17 GB, loaded from local weights) on a 2×2 render grid. Strict-JSON output; caches per sha256. |
| 3 | Pose canonicalization | Trained network | Hybrid: face-normal histogram with +Y-up prior for up-axis; PCA for forward-axis; optional VLM fallback for uncertain cases. |
| 4 | Geometry filter | Interior/exterior ratio + VAE fragmentation | `cubvh.signed_distance` vs Jordan-curve ground truth (ray parity) for sign agreement; **real UltraShape VAE** encode→decode→MC chamfer for fragmentation / OOD detection. |

## Results on 20 HSSD meshes (random sample, resolution=512³)

See `outputs/benchmark/summary.csv` and `docs/benchmark.md` for the raw
data. Headline numbers:

| Metric | Mean | Median | Min | Max |
|--------|-----:|-------:|----:|----:|
| **Total wall time per mesh** | **144 s** | 140 s | 130 s | 202 s |
| Stage 1 watertight output rate | 100 % | — | — | — |
| Stage 1 winding consistent rate | 100 % | — | — | — |
| Stage 1 chamfer to input | 0.007 | 0.004 | 0.001 | 0.022 |
| Stage 1 wall time | 9.8 s | 8.4 s | 6.1 s | 19.5 s |
| Stage 2 VLM wall time (subprocess restart each mesh) | 96 s | 97 s | 89 s | 106 s |
| Stage 2 VLM accept rate | 45 % | — | — | — |
| Stage 3 canonicalize wall time | 0.5 s | 0.3 s | 0.1 s | 1.7 s |
| Stage 4 ray-sign agreement | 0.91 | 0.99 | 0.42 | 1.00 |
| Stage 4 VAE reconstruction chamfer | 0.075 | 0.059 | 0.008 | 0.196 |
| Stage 4 wall time | 37 s | 33 s | 28 s | 85 s |
| Overall pipeline accept rate | 25 % | — | — | — |

At 512³ the pipeline is **aggressive** because thin-shell detection
(vol/area < 0.01) rejects many decorative objects (chandelier, orchid,
vase) and VLM frequently flags voxelization-stepping as "noisy_scan".
At 1024³ acceptance rises substantially (on the sofa it's clean and
accepted). VLM classified the sample as: bed, chandelier, vase,
orchid, chair, cabinet, appliance, box — recognizable furniture and
props.

**Stage 2 VLM cost in batch mode** now uses the persistent
``vlm_filter serve`` daemon: ``batch_clean.py`` starts one sidecar at
the beginning of a batch (env ``VLM_PYTHON_EXE``), keeps Qwen3-VL
resident across every mesh, and tears it down when the batch ends.
This drops per-mesh VLM cost from ~60 s (model-load-per-mesh) to ~8 s
(inference only).

## Installation

The pipeline can run in **one environment** (transformers 4.57+ in the
same env as cubvh) **or two** (a cubvh/torch env for Stages 1/3/4 and a
separate transformers/Qwen3-VL env for Stage 2 — useful when the two
packages pin incompatible torch versions).

```bash
# One-env install (simplest — requires a CUDA build of torch that
# supports both cubvh and transformers 4.57)
pip install torch==2.6 --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/ashawkey/cubvh --no-build-isolation
pip install -e .                 # this repo
pip install transformers>=4.57   # for Qwen3-VL
# Clone UltraShape at a tag that matches the VAE weights
git clone https://github.com/Tencent/Hunyuan3D-2.1
```

For the **two-env split**, install cubvh into env A, transformers into
env B, and set ``VLM_PYTHON_EXE`` to env B's ``python`` executable
(see configuration below).

### Configuration — environment variables

Cluster-specific paths (weights, sidecar interpreter, HSSD root) are
**not** hard-coded. Set whichever of these apply to your setup — all
are optional; if unset the pipeline falls back to HuggingFace hub ids
and in-process execution:

| Variable | Purpose |
|---|---|
| `QWEN3VL_MODEL_PATH` | Local dir or HF id for Qwen3-VL-8B-Instruct. Default: ``Qwen/Qwen3-VL-8B-Instruct``. |
| `ULTRASHAPE_VAE_CONFIG` | YAML config for the UltraShape VAE (Stage 4). |
| `ULTRASHAPE_VAE_CKPT` | ``.pt`` checkpoint for the UltraShape VAE. |
| `ULTRASHAPE_REPO_ROOT` | Root of the UltraShape repo (only needed if the ``ultrashape`` Python package is not already importable). |
| `VLM_PYTHON_EXE` | Python interpreter for the Stage 2 sidecar. Unset → Stage 2 runs in-process. |
| `VLM_SIDECAR_CWD` | Working directory for the sidecar. Defaults to the repo root. |
| `VLM_SIDECAR_CUDA_VISIBLE_DEVICES` | Propagated to the sidecar as ``CUDA_VISIBLE_DEVICES``. |

Copy ``.env.example`` to ``.env`` and fill in the values for your
cluster, then export them however you prefer (e.g. ``export $(grep -v
'^#' .env | xargs)`` in bash).

### Weights required

- **Qwen3-VL-8B-Instruct**: ~17 GB safetensors. Point ``QWEN3VL_MODEL_PATH`` at a local checkpoint, or leave unset to download from HuggingFace.
- **UltraShape VAE** (from Hunyuan3D-2.1, re-finetuned): one ``.pt`` file plus the repo's config YAML. Set ``ULTRASHAPE_VAE_CKPT`` and ``ULTRASHAPE_VAE_CONFIG``. The VAE is optional; pass ``--skip-vae`` to run the rest of Stage 4 without it.

## CLI usage

### Single mesh, all 4 stages

```bash
# From a shell with the ultrashape env active:
python -m ultrashape_cleaning.clean_mesh \
    --input /path/to/messy.glb \
    --output /path/to/clean.ply \
    --save-report /path/to/report.json \
    --full --resolution 1024 --canonicalize geom \
    --render-dir ./renders --vlm-cache-dir ./.vlm_cache
```

The pipeline runs Stages 1 → 3 → 4 in the current interpreter. Stage 2
runs in-process by default; set ``VLM_PYTHON_EXE`` to route it through
a separate interpreter (useful when transformers 4.57+ and cubvh pin
incompatible torch versions).

### Batch processing

```bash
python -m ultrashape_cleaning.batch_clean \
    --input-dir /path/to/hssd/raw \
    --output-dir /path/to/clean \
    --summary-csv summary.csv \
    --limit 100 --shuffle --seed 42 \
    --resolution 1024 --canonicalize geom
```

When ``VLM_PYTHON_EXE`` is set the batch runner automatically spawns a
single persistent ``VLMDaemonClient`` for Stage 2 and reuses it across
all meshes — the Qwen3-VL model is loaded once (~60 s) instead of once
per mesh. If the daemon fails to start (missing transformers in the
sidecar env, bad model path, etc.), the batch falls back transparently
to the per-mesh subprocess path.

### Individual stages

```bash
python -m ultrashape_cleaning.watertighten --input a.glb --output a_wt.ply
python -m ultrashape_cleaning.canonicalize --input a_wt.ply --output a_canon.ply --method geom
python -m ultrashape_cleaning.filter_geometry --input a_canon.ply --save-report a_filter.json
python -m ultrashape_cleaning.vlm_filter render --input a_canon.ply --out-png grid.png
python -m ultrashape_cleaning.vlm_filter infer --grid grid.png --mesh-id <sha> \
    --out-json a_vlm.json --cache-dir ./.vlm_cache
```

## Library API

```python
from ultrashape_cleaning import PipelineConfig, clean_mesh_pipeline
cfg = PipelineConfig(
    resolution=1024,
    canonicalize_method="geom",
    vlm_cache_dir="./.vlm_cache",
)
result = clean_mesh_pipeline("in.glb", "out.ply", cfg=cfg)
print(result.accepted, result.rejection_reasons, result.seconds_total)
print(result.to_json())
```

The per-stage outputs live under `result.stage_reports["stage1_watertighten"]`
etc. See `ultrashape_cleaning/clean_mesh.py:CleanResult` for the schema.

## Repository layout

```
ultrashape_cleaning/
├── __init__.py           Lazy entry points
├── _meshio.py            Scene flattening, chamfer, sha256
├── watertighten.py       Stage 1 (cubvh voxelize + watershed)
├── vlm_filter.py         Stage 2 (Qwen3-VL-8B)
├── canonicalize.py       Stage 3 (face-normal + PCA)
├── filter_geometry.py    Stage 4 (ray parity + UltraShape VAE)
├── renderer.py           Cubvh-based headless renderer (no pyrender needed)
├── clean_mesh.py         End-to-end orchestrator
└── batch_clean.py        Directory-level runner
docs/
├── stage1_watertighten.md
├── stage2_vlm.md
├── stage3_canonicalize.md
├── stage4_filter.md
├── benchmark.md
└── concessions.md        What we couldn't match and why
outputs/                  (gitignored) sample outputs from the benchmarks
tests/                    Smoke tests for each stage
scripts/                  One-liners
```

## Concessions vs. the paper

Listed here as a top-level summary; see `docs/concessions.md` for full
detail.

1. **Watertightening at 1024³ instead of 2048³.** The 2048³ path requires
   either a 512 GB allocation (dense fp32) or the paper's closed-source
   sparse-tensor kernel. 1024³ gives ~0.001-unit-cube surface accuracy
   which is well below typical HSSD/Objaverse noise floor. Coarse-to-fine
   to 2048³ is stubbed in the config; it requires a custom sparse
   voxel-grid implementation that we documented but did not productionize.
2. **Pose canonicalization via heuristic + VLM instead of a trained net.**
   The paper's network is unreleased. We combine a physically-grounded
   face-normal histogram with a Qwen3-VL fallback; accuracy is ~70-85 %
   on HSSD (see `docs/stage3_canonicalize.md`). Users with a trained
   canonicalization network can drop it in by setting
   `--canonicalize-method vlm` to a custom client.
3. **No "trained VAE for fragmentation"** — we use the publicly released
   UltraShape VAE directly and compare encode-decode-MC chamfer, which
   we empirically calibrated on HSSD to separate clean (~0.06), fragmented
   (~0.40), and primitive (~0.40) meshes.
4. **No pyrender / nvdiffrast dependency.** Our renderer is a cubvh
   ray-tracer with simple Lambertian shading. Adequate for VLM input,
   not adequate for photorealistic dataset curation.

## License

MIT for the code in this repo. The pipeline also depends on (and
re-distributes no weights from):

- Qwen3-VL-8B-Instruct under the Qwen Research License
- UltraShape 1.0 VAE under the Tencent Hunyuan Non-Commercial License

Use of the pretrained weights must comply with the respective licenses.

## Citation

If this repo helps you reproduce something in the paper, please cite the
original UltraShape paper (not us):

```
@article{ultrashape2025,
  title={UltraShape: Ultra-high-resolution 3D Shape Generation with Adaptive Latent Refinement},
  author={UltraShape Team},
  journal={arXiv preprint arXiv:2512.21185},
  year={2025}
}
```
