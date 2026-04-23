# Stage 2: VLM-based Quality Filter

## What the paper says

UltraShape §2.1 describes a vision-language model (VLM) quality filter
that renders canonical views of a mesh and asks a VLM to judge whether
the mesh is:

- A usable 3D object (accept), or
- A primitive / ground plane / noisy scan / fragmented mesh (reject).

The paper does not specify which VLM was used.

## What we ship

A **real Qwen3-VL-8B-Instruct** integration in
`ultrashape_cleaning/vlm_filter.py`. End-to-end path:

1. **Render** 4 canonical views (front / right / back / left) of the
   canonicalized mesh via our cubvh-based ray tracer
   (`ultrashape_cleaning/renderer.py`). Each view is 512×512 with
   three-light Lambertian shading and a neutral white background.
2. **Tile** the 4 views into a 2×2 grid PNG with axis labels burnt in.
3. **Prompt** Qwen3-VL-8B with a strict-JSON structured prompt asking for
   `{aesthetic_quality, is_primitive, is_ground_plane, is_noisy_scan,
   is_fragmented, object_class, reasoning}`.
4. **Parse** the JSON, coerce field types, and apply acceptance rules.
5. **Cache** results keyed by sha256 of the input mesh file in a
   `.vlm_cache` directory for cheap reruns.

### The prompt

English variant (default; Chinese variant also shipped):

```text
You are inspecting four orthogonal views of a 3D object: front (top-left),
right (top-right), back (bottom-left), left (bottom-right) of a 2x2 grid.

Please assess the object against the following criteria and return a STRICT
JSON object (no prose, no markdown fences) with these fields:

  - "aesthetic_quality": integer 1-5 ...
  - "is_primitive": true if ... plain cube/sphere/cylinder with no semantic ...
  - "is_ground_plane": true if object is essentially a flat slab ...
  - "is_noisy_scan": true if geometry looks like a noisy photogrammetric scan ...
  - "is_fragmented": true if ... disconnected pieces ...
  - "object_class": short English label (e.g. "chair", "sofa" ...)
  - "reasoning": 1-2 sentences ...

Return ONLY the JSON object.
```

The full prompts (`DEFAULT_PROMPT_EN`, `DEFAULT_PROMPT_ZH`) live at the
top of `vlm_filter.py`.

### Acceptance logic

`_accept_decision()` rejects the mesh if ANY of:

- `aesthetic_quality < min_quality` (default 2)
- `is_primitive == true` (configurable)
- `is_ground_plane == true`
- `is_noisy_scan == true`
- `is_fragmented == true`

This matches the paper's stated criteria.

### Two-env plumbing

Qwen3-VL-8B requires `transformers >= 4.57` which is incompatible with the
`torch 2.5.1 + cubvh` environment used by other stages. We solve this by
shelling Stage 2 out to a sidecar subprocess in the `buildingseg` env:

```python
# In clean_mesh.py:
cmd = [cfg.vlm_python_exe, "-m", "ultrashape_cleaning.vlm_filter", "infer",
       "--grid", grid_png, "--mesh-id", sha,
       "--out-json", vlm_json, "--model-path", cfg.vlm_model_path,
       "--prompt-lang", cfg.vlm_prompt_lang]
subprocess.run(cmd, ...)
```

The primary env does the rendering; the sidecar env does the VLM
inference. Both are non-interactive.

## Benchmark findings

Results on the 20-mesh HSSD sample set (see `docs/benchmark.md` for the
full numbers). Representative examples:

| Mesh | VLM class | quality | accepted? | rationale |
|------|-----------|--------:|-----------|-----------|
| sofa (00366b…) | "bench" | 4 | yes | VLM correctly identifies seating furniture |
| grey clock (00258b…) | "clock" | 4 | yes | Cylindrical shape with legs; correctly recognized |
| antique bottle (00386b…) | "vase" | 3 | yes | Close category; accepted |
| (primitive cube, synthetic) | "unidentifiable" | 1 | no | is_primitive=true |

Qwen3-VL-8B takes **~7-10 seconds per image** on an A100-80GB once
loaded. First-load is ~60s (4 shards, bf16, ~17 GB). The cache means
re-running is **~0 s** after the first pass.

## Latency budget

- Model load (first mesh only): 60-90 s
- Per-mesh inference: 7-10 s
- Render: 2-3 s
- Subprocess overhead: +5 s (loading transformers + tokenizer each call)

Mitigation: in `batch_clean.py` the pipeline keeps a **persistent VLM
server** resident in the sidecar process if you set
`cfg.vlm_python_exe=None` and run the whole pipeline under the
`buildingseg` env (which also needs cubvh + trimesh — currently the
easiest setup is to install cubvh into buildingseg or install
transformers 4.57 into ultrashape).

## Concessions vs the paper

1. **Which VLM** — the paper doesn't specify. Qwen3-VL-8B is a strong
   choice: open-weights, robust on structured JSON, good zero-shot object
   recognition. Qwen3-VL-32B is available at
   `/moganshan/afs_a/lbx/hf/hub/models--Qwen--Qwen3-VL-32B-Instruct/`
   and can be swapped in by passing `--model-path`.

2. **No fine-tuning.** The paper hints at fine-tuned VLMs for quality
   scoring. We use zero-shot prompting, which works well in practice but
   occasionally mislabels (e.g. our outdoor sofa was classified as a
   "bench" — not incorrect, just a different ontology node).

3. **Photorealism.** Our shaded renders are Lambertian-only with no
   shadows/textures. The paper's VLM may have seen richer renders. On our
   test set this hasn't caused false positives — a sofa is a sofa even
   in flat shading.
