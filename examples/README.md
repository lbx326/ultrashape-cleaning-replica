# Example outputs

Renders from the pipeline on HSSD samples. These are the 2×2 grid PNGs
used as VLM input (front / right / back / left).

## `sofa_raw_4view.png`

HSSD mesh `00366b86401aa16b702c21de49fd59b75ab9c57b.glb` rendered directly
from the GLB (without any cleaning). The outdoor sofa is already in a
reasonable orientation (Y-up) thanks to the GLB convention, but is
slightly mis-aligned on the front axis — the "front" camera sees the
long side of the sofa, not the face-the-viewer front.

Qwen3-VL correctly classifies this as a "sofa"/"bench" with aesthetic
quality 4 and accepts it.

## `sofa_canonicalized_4view.png`

The same mesh after Stage 1 (watertighten at 1024³) + Stage 3
(canonicalize with the `geom` method). Now the "front" view shows the
short face (where you actually sit facing), and the legs are correctly
at the bottom.

## `bed_vlm_rejected_4view.png`

HSSD mesh `75268284ec78cf6f55a1e2c715e01e1f3cd466b9111d6cd7c6f7d103d9b91371.glb`
(a bed with ribbed comforter) after cleaning at 512³. Because the 512³
voxelization shows horizontal stepping on the comforter, Qwen3-VL
classifies this as "bed" with `is_noisy_scan=true` and **rejects** it.

This is a genuine false positive of our filter at 512³; at 1024³ the
ribbing would be smoother and the VLM would likely accept the mesh.
This illustrates the trade-off between Stage 1 resolution and downstream
acceptance rates.

## `chair_accepted_4view.png`

Sample #3 from the 20-mesh benchmark. HSSD mesh `2a7342c1f2…` (a
wall-mounted shelf/chair structure) after the full pipeline:

- Stage 1 watertight: ✓ (1.16M faces out)
- Stage 3 canonicalize: correct +Y up
- Stage 4 ray agreement: 0.995, VAE chamfer 0.084 (well below 0.15 threshold)
- Stage 2 VLM: "chair", quality 3, **accepted**

## `appliance_accepted_4view.png`

Sample #5 from the benchmark. A voxelized household appliance:

- Qwen3-VL correctly identifies this as "appliance", quality 4
- VAE chamfer 0.057 (clean)
- Accepted.
