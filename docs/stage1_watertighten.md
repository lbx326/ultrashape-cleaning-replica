# Stage 1: Watertightening

## What the paper says

UltraShape §2.1 describes a CUDA-parallel sparse-voxel reconstruction
that processes meshes at **2048³ effective resolution** on GPU. The
algorithm it sketches is:

1. Voxelize triangles at 2048³ into a sparse occupancy volume.
2. Apply **watershed-inspired hole closing**: flood-fill from known-outside
   voxels, labeling voxels by "basin" membership; close holes where the
   watershed boundary closes within a learned epsilon.
3. **Volumetric thickening** for open surfaces: detect open shells via
   ray-escape ratios, thicken along outward face normals adaptively.
4. Build a signed-distance field.
5. Extract isosurface via marching cubes.

The implementation is closed-source.

## What we ship

A three-backend pipeline living in `ultrashape_cleaning/watertighten.py`.
The main algorithm path is:

```
triangle voxelization (GPU, cubvh distance query)
    |
    v
morphological closing (GPU, max-pool based 26-connected dilation + erosion)
    |
    v
watershed-inspired flood-fill from grid boundary
    |
    v
open-shell detection (ray escape fraction from shell voxels)
    |
    v
(if open) adaptive thickening -> re-flood
    |
    v
signed distance field (cubvh.signed_distance + flood-fill sign correction)
    |
    v
marching cubes (scikit-image)
    |
    v
largest-component filter + normal/winding fix
```

### Concrete technical choices

- **Voxelization**: `voxelize_shell_cubvh` queries `cubvh.unsigned_distance`
  at every voxel center and thresholds at `band_voxels / R` voxel widths.
  This is equivalent to the exact triangle-voxel intersection test but
  runs entirely on GPU. We chunk the queries to 4M points at a time to
  cap VRAM; at 1024³ this peaks at about **24 GB** on an A100-80GB.
  The 2048³ path needs ~192 GB peak and is gated behind a future
  coarse-to-fine ROI refinement (not yet productionized — see
  "Concessions" below).

- **Watershed-like flood-fill**: We implement the flood as iterative
  26-connected dilation using `F.max_pool3d(3, stride=1, padding=1)`.
  Each iteration extends the "outside" region by one voxel, masked by
  the shell. Convergence is detected via count of outside voxels every
  16 iterations. On the sofa benchmark this takes **about 500-600
  iterations** (diagonal crossings of the 1024³ grid) and completes in
  36 seconds.

- **Open-shell detection**: When `interior_frac_raw < 0.5 %` we run
  `_ray_escape_fraction`: from shell voxel centers, shoot random rays
  via `cubvh.ray_trace`; the escape fraction tells us how leaky the
  shell is. Above 40 % we mark the mesh as an open shell and run
  adaptive thickening.

- **Adaptive thickening**: `_gpu_binary_dilation` on the closed shell by
  `thicken_voxels` steps. This is CUDA max-pool-based 26-connected
  dilation, matching the closing kernel. Re-flood after thickening to
  get a new occupancy volume.

- **Signed distance**: `cubvh.signed_distance` gives us true geodesic
  distance with signs depending on mesh winding. For meshes with bad
  winding (common in Hunyuan3D output), the sign is unreliable so we
  compute `|SDF|` from cubvh and take the sign from the flood-fill
  occupancy. This guarantees a consistent sign field.

- **Marching cubes**: `skimage.measure.marching_cubes(sdf_np, level=0)`.
  We don't use `cubvh`'s MC because it requires specific layouts; the
  skimage version is proven and fast enough (~15s at 1024³ on CPU).

## Benchmarks

Measured on HSSD `00366b86401aa16b702c21de49fd59b75ab9c57b.glb` (sofa,
35k faces in, ~5.7M out at 1024³). See `outputs/benchmark/summary.csv`
for the 20-mesh set.

| Resolution | Total wall | Voxelize | Flood-fill | SDF | MC | VRAM peak | Watertight |
|-----------:|-----------:|---------:|-----------:|----:|---:|----------:|------------|
|      256³  |      8.5 s |    5.6 s |      0.5 s | 0.5 s | 0.3 s |   <2 GB | **True** |
|      512³  |      ~25 s |      ~7 s |     ~7 s  |   2 s |   4 s |    4 GB | True |
|     1024³  |       90 s |      5.2 s |     36.4 s | 7.5 s | 16.4 s |   24 GB | **True** |

At 1024³ the chamfer-to-input distance is ~0.011 (unit cube) which is
below typical HSSD triangulation error. Good enough for downstream
UltraShape training pipelines that quantize to 409600 surface samples
anyway.

## Concessions vs the paper

1. **No 2048³ dense path.** Dense fp32 at 2048³ is 32 GB per volume;
   running cubvh distance queries on 8.5 billion points in 4M-point
   chunks would take ~15 minutes per mesh. A sparse-tensor + hash-map
   representation would be required to match the paper's throughput.
   We stub this as `WatertightenConfig.coarse_to_fine=True` with
   resolution=2048 and dense_resolution=512; currently not implemented.

2. **Watershed inspired, not exact.** The paper's description ("watershed-
   inspired") doesn't commit to a specific segmentation algorithm. We
   implement the flood-fill interpretation where the "watershed boundary"
   is the outside/inside transition. We do not label multiple basins.
   Objects with toroidal topology work fine because MC handles arbitrary
   genus; objects with multiple *exterior* cavities (e.g. hollow
   bookshelf) retain their cavities because flood from grid boundary
   reaches them.

3. **Ray escape is coarse.** We sample 256 random shell-voxel origins and
   cast one ray per. The paper may do something more principled (e.g.
   compute solid-angle integration). Our threshold of 0.4 escape fraction
   is empirical — see `docs/concessions.md`.

## File map

- `ultrashape_cleaning/watertighten.py` — all of the above
- `ultrashape_cleaning/_meshio.py` — scene flatten, fit-to-unit-cube,
  chamfer distance, summary helpers
- `tests/test_watertighten.py` — holed-box and sphere sanity tests
