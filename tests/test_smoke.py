"""Smoke tests covering import surface + synthetic mesh sanity.

Run with:

    python tests/test_smoke.py

Or with pytest if installed:
    pytest tests/test_smoke.py -v

CUDA-dependent tests are skipped when ``torch.cuda.is_available()`` is
false, so CPU-only clones of the repo can still exercise the import
surface and the pure-Python checks.

Each test prints PASS/FAIL and returns a nonzero exit code on failure.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def make_holed_sphere() -> trimesh.Trimesh:
    """A slightly-broken sphere: remove one face to create a hole."""
    m = trimesh.creation.icosphere(radius=0.5, subdivisions=3)
    faces = m.faces.tolist()
    faces.pop(0)  # create a small hole
    return trimesh.Trimesh(vertices=m.vertices, faces=np.array(faces),
                           process=False)


def make_unit_cube() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(1.0, 1.0, 1.0))


def make_two_spheres(gap: float = 1.5) -> trimesh.Trimesh:
    """Two disjoint icospheres — fragmented."""
    a = trimesh.creation.icosphere(radius=0.1)
    b = trimesh.creation.icosphere(radius=0.1)
    b.vertices += np.array([gap, 0, 0])
    return trimesh.util.concatenate([a, b])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_import():
    from ultrashape_cleaning import PipelineConfig
    assert PipelineConfig is not None
    print("[PASS] test_import")


def test_meshio_roundtrip():
    from ultrashape_cleaning._meshio import (load_mesh, save_mesh,
                                              sha256_file, summarize,
                                              fit_to_unit_cube)
    cube = make_unit_cube()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cube.ply"
        save_mesh(cube, p)
        assert p.exists()
        sha = sha256_file(p)
        assert len(sha) == 64
        loaded = load_mesh(p)
        assert len(loaded.vertices) == len(cube.vertices)
        s = summarize(loaded)
        assert s["is_watertight"] is True
        fit, center, scale = fit_to_unit_cube(loaded)
        assert fit.vertices.max() <= 1.0
        assert fit.vertices.min() >= 0.0
    print("[PASS] test_meshio_roundtrip")


def test_watertighten_sphere():
    """A holed sphere should come out watertight after Stage 1."""
    try:
        import torch
    except ImportError:
        print("[SKIP] test_watertighten_sphere (torch not installed)")
        return
    if not torch.cuda.is_available():
        print("[SKIP] test_watertighten_sphere (no CUDA)")
        return
    from ultrashape_cleaning.watertighten import (WatertightenConfig,
                                                   watertighten_mesh)
    m = make_holed_sphere()
    assert not m.is_watertight, "input should have a hole"
    cfg = WatertightenConfig(resolution=128, dense_resolution=128,
                             close_iters=1)
    out, sdf, rep = watertighten_mesh(m, cfg, verbose=False)
    assert rep.is_watertight, f"output should be watertight, got {rep}"
    assert rep.num_faces_out > 100
    print(f"[PASS] test_watertighten_sphere "
          f"(V={len(out.vertices)}, F={len(out.faces)}, "
          f"chamfer={rep.chamfer_to_input:.4f})")


def test_renderer_outputs_rgb():
    try:
        import torch
    except ImportError:
        print("[SKIP] test_renderer_outputs_rgb (torch not installed)")
        return
    if not torch.cuda.is_available():
        print("[SKIP] test_renderer_outputs_rgb (no CUDA)")
        return
    from ultrashape_cleaning.renderer import render_four_views, make_2x2_grid
    m = trimesh.creation.icosphere(radius=0.4, subdivisions=2)
    views = render_four_views(m, resolution=128)
    assert set(views.keys()) == {"front", "right", "back", "left"}
    for k, im in views.items():
        assert im.shape == (128, 128, 3)
        assert im.dtype.name == "uint8"
    grid = make_2x2_grid(views)
    assert grid.shape == (256, 256, 3)
    print("[PASS] test_renderer_outputs_rgb")


def test_canonicalize_geom():
    from ultrashape_cleaning.canonicalize import canonicalize_mesh
    # Generate a plane-like object: box extruded along Y
    m = trimesh.creation.box(extents=(2.0, 0.5, 1.0))
    out, R, rep = canonicalize_mesh(m, method="geom")
    assert rep.method == "geom"
    # The returned rotation should be orthogonal.
    R = np.asarray(R)
    err = float(np.linalg.norm(R @ R.T - np.eye(3)))
    assert err < 1e-5, f"R not orthogonal: err={err}"
    # Determinant MUST be +1 (proper rotation); a reflection (-1) would
    # mirror the object instead of rotating it. ``_build_rotation``
    # flips the ``right`` axis when the input frame is left-handed to
    # enforce this; see canonicalize.py::_build_rotation.
    det = float(np.linalg.det(R))
    assert abs(det - 1.0) < 1e-5, f"R must be proper rotation, got det={det}"
    print("[PASS] test_canonicalize_geom")


def test_filter_geometry_no_vae():
    try:
        import torch
    except ImportError:
        print("[SKIP] test_filter_geometry_no_vae (torch not installed)")
        return
    if not torch.cuda.is_available():
        print("[SKIP] test_filter_geometry_no_vae (no CUDA)")
        return
    from ultrashape_cleaning.filter_geometry import (filter_geometry,
                                                      FilterConfig)
    m = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    cfg = FilterConfig(skip_vae=True, ray_samples=2000)
    rep = filter_geometry(m, cfg=cfg)
    # Cube has 8 vertices -> triggers primitive detection.
    assert "primitive_like" in rep.reasons, f"got reasons={rep.reasons}"
    # Ray sign agreement should be near 1.0 on a perfect cube.
    assert rep.metrics.get("ray_sign_agreement", 0) > 0.9, rep.metrics
    print(f"[PASS] test_filter_geometry_no_vae reasons={rep.reasons}")


def test_vlm_response_parsing():
    from ultrashape_cleaning.vlm_filter import parse_vlm_response
    raw = ('```json\n{"aesthetic_quality": 4, "is_primitive": false, '
           '"object_class": "chair", "reasoning": "ok", "is_ground_plane": '
           '"false", "is_noisy_scan": false, "is_fragmented": false}\n```')
    parsed = parse_vlm_response(raw)
    assert parsed["aesthetic_quality"] == 4
    assert parsed["is_primitive"] is False
    assert parsed["is_ground_plane"] is False  # coerced from string "false"
    assert parsed["object_class"] == "chair"
    # Degenerate case.
    bad = "no json here, completely unstructured"
    parsed = parse_vlm_response(bad)
    assert parsed["aesthetic_quality"] is None
    assert parsed["_parse_error"] is True
    print("[PASS] test_vlm_response_parsing")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def main() -> int:
    tests = [
        test_import,
        test_meshio_roundtrip,
        test_watertighten_sphere,
        test_renderer_outputs_rgb,
        test_canonicalize_geom,
        test_filter_geometry_no_vae,
        test_vlm_response_parsing,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"[FAIL] {t.__name__}: {e}")
        except Exception as e:
            failures += 1
            import traceback
            traceback.print_exc()
            print(f"[ERROR] {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
