"""_meshio.py -- Shared mesh IO helpers used by every stage.

We consolidate the load / save / sha256 logic here so all stages produce
consistent canonical Trimesh objects. HSSD GLBs arrive as multi-node Scenes
with per-node transforms; we collapse them into a single mesh at load time.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import trimesh


PathLike = Union[str, Path]


def sha256_file(path: PathLike, chunk: int = 1 << 20) -> str:
    """Stream sha256 of a file on disk."""
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def _flatten_scene_to_mesh(obj) -> trimesh.Trimesh:
    """Flatten a trimesh.Scene (with per-node transforms) into one Trimesh.

    Handles:
    - Scene graph transforms (dump() baked)
    - Non-Trimesh geometry (PointCloud / Path3D) — skipped
    - Empty scenes — raises RuntimeError

    We prefer ``Scene.dump(concatenate=True)`` which trimesh 3.x/4.x
    implements as "bake each geometry by its scene-graph transform and
    concatenate". Falling back to manual per-geometry transform lookup
    if that ever stops being available.
    """
    if isinstance(obj, trimesh.Trimesh):
        return obj
    if isinstance(obj, trimesh.Scene):
        # Fast path: let trimesh do the graph-bake + concat itself.
        try:
            dumped = obj.dump(concatenate=True)
            if isinstance(dumped, trimesh.Trimesh) and len(dumped.faces) > 0:
                return dumped
        except Exception:
            pass
        # Fallback: iterate geometries manually.
        meshes: list[trimesh.Trimesh] = []
        for name, geom in obj.geometry.items():
            if not isinstance(geom, trimesh.Trimesh):
                continue
            T = np.eye(4)
            try:
                T = obj.graph.get(name)[0]  # world -> geometry transform
            except Exception:
                # Some scenes don't have "world" in the graph under that name;
                # search for a node whose geometry matches.
                for node_name in obj.graph.nodes_geometry:
                    try:
                        geom_name = obj.graph[node_name][1]
                        if geom_name == name:
                            T = obj.graph.get(node_name)[0]
                            break
                    except Exception:
                        continue
            baked = geom.copy()
            baked.apply_transform(T)
            meshes.append(baked)
        if not meshes:
            raise RuntimeError("Scene contains no Trimesh geometry")
        if len(meshes) == 1:
            return meshes[0]
        return trimesh.util.concatenate(meshes)
    raise RuntimeError(f"Cannot flatten {type(obj)} into Trimesh")


def load_mesh(path: PathLike, process: bool = False) -> trimesh.Trimesh:
    """Load any mesh format as a single Trimesh.

    ``process=False`` is important on input because we want to preserve
    degenerate-face artifacts so stage 1 can see and voxel-over them.
    Internal callers may re-call with process=True after stage 1.
    """
    obj = trimesh.load(str(path), process=process, force=None)
    mesh = _flatten_scene_to_mesh(obj)
    # Strip any vertex colors / UVs that will confuse downstream stages.
    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
    )
    if len(mesh.faces) == 0:
        raise RuntimeError(f"{path}: mesh has zero faces")
    return mesh


def save_mesh(mesh: trimesh.Trimesh, path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))


def fit_to_unit_cube(
    mesh: trimesh.Trimesh, pad: float = 0.03
) -> tuple[trimesh.Trimesh, np.ndarray, float]:
    """Center mesh at 0.5 and scale so the bbox sits in [pad, 1-pad]^3.

    Returns (mesh_fit, center_world, scale). To invert:
        verts_world = (verts_fit - 0.5) * scale + center_world
    """
    v = np.asarray(mesh.vertices, dtype=np.float64)
    vmin, vmax = v.min(0), v.max(0)
    center = (vmin + vmax) * 0.5
    extent = float((vmax - vmin).max())
    if extent <= 0:
        raise RuntimeError("Mesh has zero extent")
    scale = extent * (1.0 + 2.0 * pad)
    v_fit = (v - center) / scale + 0.5
    return (
        trimesh.Trimesh(vertices=v_fit, faces=mesh.faces.copy(), process=False),
        center,
        scale,
    )


def largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Keep only the largest connected component by face count."""
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    sizes = [len(p.faces) for p in parts]
    return parts[int(np.argmax(sizes))]


def decimate_if_huge(mesh: trimesh.Trimesh, max_faces: int = 1_000_000
                    ) -> tuple[trimesh.Trimesh, bool]:
    """If mesh has more than max_faces, run trimesh quadric decimation."""
    if len(mesh.faces) <= max_faces:
        return mesh, False
    try:
        target = min(max_faces, len(mesh.faces) // 2)
        simple = mesh.simplify_quadric_decimation(face_count=target)
        if simple is None or len(simple.faces) == 0:
            return mesh, False
        return simple, True
    except Exception:
        return mesh, False


def chamfer_distance(
    mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, n_samples: int = 50_000
) -> float:
    """Symmetric chamfer distance between two meshes via surface sampling."""
    # Sample N points uniformly on each mesh's surface.
    pts_a = mesh_a.sample(n_samples)
    pts_b = mesh_b.sample(n_samples)
    # a->b
    from scipy.spatial import cKDTree
    tree_b = cKDTree(pts_b)
    d_ab, _ = tree_b.query(pts_a, k=1)
    tree_a = cKDTree(pts_a)
    d_ba, _ = tree_a.query(pts_b, k=1)
    return float(d_ab.mean() + d_ba.mean()) * 0.5


def summarize(mesh: trimesh.Trimesh) -> dict:
    """Compact summary dict for reports."""
    return {
        "num_vertices": int(len(mesh.vertices)),
        "num_faces": int(len(mesh.faces)),
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(mesh.euler_number) if mesh.is_watertight else None,
        "volume": float(mesh.volume) if mesh.is_watertight else None,
        "area": float(mesh.area),
        "bbox_extent": [float(x) for x in mesh.extents],
    }
