# Changelog

All notable changes to this repository are documented here.

## [Unreleased] — 2026-04-23

### Fixed per external code review

An external sibling agent audited the pipeline against the paper and
against a general "clonable from GitHub" standard; findings in
``D:/Claude/mesh_clean_tests/REVIEW.md``. The following items from that
review have landed:

- **P0**: `watertighten.py` no longer calls the `remove_duplicate_faces()`
  method removed in trimesh 4.x. Uses ``update_faces(unique_faces())``
  with a graceful fallback for pre-4.0 trimesh.
- **P0**: `watertighten.py`'s verbose post-Stage-1 print no longer
  raises ``TypeError`` when the chamfer distance could not be computed.
  Split the conditional expression into an explicit ``if``.
- **P0**: Cluster-specific absolute paths
  (``/moganshan/afs_a/...``) removed from every source file, the demo
  script, and the test suite. All paths now resolve via
  ``ultrashape_cleaning/_config.py`` (`QWEN3VL_MODEL_PATH`,
  `ULTRASHAPE_VAE_CONFIG`, `ULTRASHAPE_VAE_CKPT`,
  `ULTRASHAPE_REPO_ROOT`, `VLM_PYTHON_EXE`, `VLM_SIDECAR_CWD`,
  `VLM_SIDECAR_CUDA_VISIBLE_DEVICES`) — see ``.env.example``.
- **P0**: ``VLMDaemonClient`` is now actually wired in.
  ``batch_clean.batch_clean`` starts a single persistent Qwen3-VL
  subprocess at the beginning of a batch and reuses it across every
  mesh, cutting per-mesh Stage-2 wall-clock from ~60 s (model-load
  per mesh) to ~8 s (inference only) whenever ``VLM_PYTHON_EXE`` is
  configured. Transparent fallback to per-mesh sidecar if the daemon
  fails to start.
- **P1**: ``filter_geometry.py::encode_decode_chamfer`` maps marching-
  cubes vertices to voxel-center world coordinates. The previous code
  assumed a corner grid and biased the reconstruction by ~1/query_grid
  in each axis (~0.008 at grid=128), inflating chamfer calibration.
- **P1**: ``watertighten.py::_ray_escape_fraction`` seeds origins from
  true interior voxels (``inside_raw``) instead of shell voxels; rays
  from shell cells hit the shell immediately and biased the escape
  rate low.
- **P1**: ``canonicalize.py::_build_rotation`` now guarantees a proper
  rotation (``det(R) = +1``). If the input frame is left-handed the
  ``right`` axis is flipped so the output never mirrors the object.
  Smoke test updated to enforce ``det == +1`` (previously tolerated
  ``|det| == 1``).
- **P2**: Honest documentation. The ``watertighten.py`` module docstring
  claimed "sparse COO tensor" storage; the implementation is and has
  always been dense fp32 at 1024³. Docstring, ``docs/concessions.md``
  §1, and ``docs/stage1_watertighten.md`` updated to say so. The
  ``coarse_to_fine`` config flag is now labelled a reserved no-op
  rather than a partially-implemented path.

### Added

- ``ultrashape_cleaning/_config.py`` — central resolver for
  cluster-specific paths via environment variables.
- ``.env.example`` — template for local configuration.
- README "Configuration — environment variables" section.
