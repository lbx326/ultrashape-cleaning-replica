# Concessions vs. the paper (full detail)

This is the honest list of where our open-source replica falls short of
UltraShape 1.0's closed-source data-cleaning pipeline, and why.

## 1. Watertightening — 1024³ dense, not 2048³ sparse

**Paper**: 2048³ effective resolution via a custom CUDA sparse-tensor
kernel.

**Ours**: 1024³ effective resolution via `cubvh.unsigned_distance` +
**dense fp32** occupancy volume. The implementation is dense at every
stage (shell, closed shell, outside, occupancy, SDF). The module
docstring previously mentioned "sparse COO tensor" storage — that was
aspirational wording and did not match the code; the docstring has been
corrected.

**Why we can't match**:
- A dense fp32 volume at 2048³ is **32 GB** per copy, and the pipeline
  needs at least 3 copies (shell, occupancy, SDF). Even at fp16 this is
  48 GB peak, barely fitting on A100-80GB alongside cubvh's BVH.
- A sparse-tensor implementation would require either `fVDB` (NVIDIA
  sparse voxel library, not installed) or `OpenVDB` Python bindings
  (not installed, pip install is nontrivial with CUDA).
- Writing a custom sparse 3D watershed kernel in pure PyTorch is
  possible but was judged out of scope for this reimplementation.

**Impact**: surface accuracy at 1024³ is ~0.0005 units (a few mm on a
1-meter object). This is well below the typical 0.002-0.005 noise floor
of HSSD / Objaverse GLBs, so the quality gap is small in practice.

**Future work**: coarse-to-fine ROI refinement. Run 512³ dense flood-
fill for overall occupancy; then run `cubvh.signed_distance` at 2048³
only within 4 voxels of the shell (`sdf_band_voxels=4`); marching cubes
operates on the sparse SDF band. ``WatertightenConfig.coarse_to_fine``
is reserved for this; the flag exists but the current code does **not**
branch on it, i.e. it is a **no-op**, not an "opt-in" path.

## 2. Watershed hole closing — simplified to dilation-based

**Paper**: "watershed-inspired" hole closing. Watershed segmentation
(Meyer / Vincent / Soille formulations) labels basins and finds ridge
lines; hole-closing fills any region whose basin boundary closes.

**Ours**: straight dilation (morphological closing) of the shell, then
flood-fill from the grid boundary. Multi-basin labeling is NOT
implemented.

**Impact**: we cannot distinguish legitimate cavities (hollow vase) from
holes-to-be-closed (small punctures). Our closing is size-thresholded:
holes smaller than `close_iters=2` voxels are filled; larger cavities
are preserved. This works well in practice for HSSD.

**Future work**: scikit-image's `segmentation.watershed` on the
unsigned-distance field would give us basin labels and a principled
hole-size threshold.

## 3. Pose canonicalization — heuristic, not learned

**Paper**: trained neural network (architecture + dataset unspecified).

**Ours**: face-normal histogram + PCA + VLM fallback.

**Why we can't match**: training data (canonical orientation pairs) not
released; retraining from scratch on Objaverse or ShapeNet would require
~40-80 GPU-hours and is out of scope.

**Impact**: our `geom` method hits ~65% accuracy on HSSD; the `hybrid`
method with Qwen3-VL hits ~80-85%. The paper's network almost certainly
exceeds 95%.

**Future work**: wrap the Hunyuan3D-2.1 canonicalization net (if
available in that repo's weights — we didn't locate one) or train our
own.

## 4. Fragmentation VAE — we reuse UltraShape's, paper had a dedicated one

**Paper**: reads as if a separate VAE was trained specifically for
out-of-distribution detection on meshes.

**Ours**: load the publicly released UltraShape VAE and measure encode-
decode-MC chamfer. A mesh far from the VAE's training manifold (clean
indoor objects) will reconstruct poorly.

**Impact**: our threshold (0.15 chamfer) is calibrated on sofas vs
synthetic primitives. On boundary cases — e.g. a clean but unusual mesh
like an abstract sculpture — we may false-positive.

## 5. Renderer — no pyrender / nvdiffrast

**Paper**: doesn't specify. Most published 3D pipelines use pyrender or
nvdiffrast for VLM-input rendering.

**Ours**: a ~200 LOC cubvh-based ray tracer with Lambertian shading
(key + fill + ambient). No shadows, no textures, no specular.

**Impact**: VLM renders are clean but flat. Qwen3-VL still correctly
identifies sofas, chairs, clocks, bottles in our testing — the shape
recognition doesn't require photorealism.

## 6. Two-env split for VLM

The Qwen3-VL-8B model requires `transformers >= 4.57` for native
support (`Qwen3VLForConditionalGeneration`), which in turn wants
`torch >= 2.5`. But `cubvh` on the reference cluster is built against
`torch 2.5.1` in one venv, and `transformers 4.57` is in another venv
with `torch 2.6.0`. We bridge this by running Stage 2 as a subprocess:

```python
subprocess.run([cfg.vlm_python_exe, "-m", "ultrashape_cleaning.vlm_filter",
                "infer", ...])
```

For **single-mesh** runs (`clean_mesh.py`) the subprocess reloads the
model each time, costing ~60 s per call. For **batch** runs
(`batch_clean.py`) we now spawn a single persistent ``VLMDaemonClient``
at the start of the batch (env var ``VLM_PYTHON_EXE``), keep the model
resident across all meshes, and tear it down when the batch ends. This
drops per-mesh VLM cost to ~8 s. The daemon speaks a JSONL protocol
defined in ``vlm_filter.py::_cli_serve``; see ``batch_clean.py``
(``batch_clean`` function) for the plumbing.

## 7. No training-distribution calibration

We calibrated the chamfer threshold on 3 meshes (clean sofa + synthetic
cube + two-sphere fragment). A proper calibration would compute the
distribution over hundreds of "good" HSSD meshes and pick a threshold
at, say, the 95th percentile. Left as future work.
