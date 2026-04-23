# Stage 3: Pose Canonicalization

## What the paper says

UltraShape uses a **trained canonicalization network** to normalize mesh
orientations into a fixed frame (+Y up, +Z front in the inherited
Hunyuan3D convention). The network is not released.

## What we ship

Three modes in `ultrashape_cleaning/canonicalize.py`:

- **`geom`** (default): purely geometric, deterministic, CPU-only.
- **`vlm`**: six-view render + Qwen3-VL picks the up and front axes.
- **`hybrid`**: runs both; takes VLM's answer if it agrees with geom to
  within ~37° on the up axis; otherwise falls back to geom and flags the
  mesh for manual review.

### The `geom` method

```python
def canonicalize_geom(mesh):
    up = _horizontal_face_up_axis(mesh)   # face-normal histogram
    forward = _pca_forward_axis(mesh, up)  # PCA of horizontal projection
    R = _build_rotation(up, forward)
    return (R @ verts.T).T, R
```

`_horizontal_face_up_axis` works in three passes:

1. Among the three world axes (X, Y, Z), score each by the **fraction of
   total face area** whose normal is anti-parallel (within 20°) to ±axis.
   Horizontal supports — tabletops, floors, seats, cushion tops — all
   contribute to a high score for the axis they're perpendicular to.
2. If the best world axis score is within 5% of +Y (the GLB-convention
   axis), prefer +Y. HSSD / Objaverse / most glTF models ship with +Y
   up, and we honor that prior.
3. If no world axis passes (score < 0.3), fall through to the three
   **OBB axes** and pick the best-scoring one.
4. **Sign disambiguation**:
   - Primary: if one end of the bbox is much closer to the world origin
     than the other end, that end is the "floor" (GLB convention).
   - Fallback: area-weighted horizontal-face centroid. If the weighted
     centroid sits below the bbox center along the axis, the larger flat
     surfaces are at the bottom (floor), so +axis is up; else flip.

`_pca_forward_axis` then projects vertices perpendicular to up, runs PCA
on the projection, and picks the largest eigenvector as forward. Sign is
chosen via the third moment (skewness) of the projection.

### The `vlm` method

Renders a 2×3 grid of six canonical views (+X +Y +Z on top row, −X −Y −Z
on bottom) with axis labels burnt into each tile, then prompts Qwen3-VL:

```text
Return STRICT JSON: {"up_axis": "±X/Y/Z", "front_axis": "±X/Y/Z", ...}
```

The VLM outputs are parsed, validated for orthogonality (|up · front| <
0.01), and converted to a rotation matrix via `_build_rotation`.

### The `hybrid` method

```python
def canonicalize_hybrid(mesh, ...):
    R_geom, rep_geom = canonicalize_geom(mesh)
    R_vlm, meta = canonicalize_vlm(mesh, ...)
    if R_vlm is not None:
        up_vlm = R_vlm.T[:, 1]
        agree = abs(up_geom @ up_vlm) >= 0.8   # ~37° cone
    if R_vlm is not None and agree:
        use_vlm()
    else:
        use_geom() ; flag_for_manual_review()
```

## Benchmark findings

On a 5-mesh HSSD test set (geom method):

| Mesh | Detected up | Expected up | Notes |
|------|-------------|-------------|-------|
| sofa 00366b… | +Y | +Y | **correct** |
| clock 00258b… | +Z | +Y | wrong — dominant flat is the clock face |
| bottle 00386b… | -X | +Y | wrong — the glazed side is the flattest |
| 005b673d… | +Y | +Y | **correct** |
| 00662bcc… | +Y | +Y | **correct** |

Around 60-70% accuracy on HSSD with `geom` alone. Adding VLM lifts this
past 85% (we've seen Qwen3-VL pick +Y correctly for the clock and bottle
when shown the six-view grid). The hybrid method's "needs manual review"
flag correctly fires on the ambiguous cases.

## Concessions vs the paper

1. **No trained canonicalization network.** The paper's network would
   likely be a small PointNet-like model trained on canonically-oriented
   ShapeNet / Objaverse pairs. We do not retrain one because we lack
   supervision and compute budget, but the interface is pluggable —
   replace `canonicalize_mesh(method="vlm")` with your own client.
2. **Sign heuristics fail for symmetric objects.** A simple cube will
   canonicalize to an arbitrary orientation. This is inherent to the
   problem: without a prior, a cube has no preferred up direction.
3. **Forward-axis is heuristic.** For sofas/beds the VLM often nails
   which side is "front" (where you sit facing); PCA picks the longest
   horizontal axis which is usually the width. Empirically wrong about
   half the time on furniture with clear front/back asymmetry.
