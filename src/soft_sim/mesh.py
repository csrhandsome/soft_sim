"""STL loading and reusable tetrahedral geometry. No simulator imports here."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from .config import resolve_asset


def load_stl(path, scale=0.001):
    raw = Path(path).read_bytes()
    count = struct.unpack_from('<I', raw, 80)[0] if len(raw) >= 84 else 0
    if len(raw) == 84 + count * 50:
        dtype = np.dtype([('normal', '<f4', (3,)), ('v', '<f4', (3, 3)), ('attr', '<u2')])
        triangles = np.frombuffer(raw, dtype=dtype, offset=84)['v'].astype(float)
    else:
        coordinates = [list(map(float, line.split()[1:])) for line in raw.decode().splitlines()
                       if line.strip().startswith('vertex ')]
        triangles = np.asarray(coordinates).reshape(-1, 3, 3)
    if not len(triangles) or not np.isfinite(triangles).all():
        raise ValueError("STL is empty or contains non-finite vertices")
    vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    if not np.all(counts == 2):
        raise ValueError("STL must be watertight: every surface edge must belong to two faces")
    return vertices * scale, faces


def refine_surface(vertices, faces, max_edge_m):
    """Split long edges without moving the STL surface or opening T-junctions.

    Midpoints are shared by both incident triangles. Unlike shading subdivision,
    this adds geometric vertices that can follow a bending FEM.
    A zero limit preserves the original topology for baseline comparisons.
    """
    if not np.isfinite(max_edge_m) or max_edge_m < 0:
        raise ValueError("Surface edge limit must be finite and nonnegative")
    vertices, faces = np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)
    if max_edge_m == 0:
        return vertices.copy(), faces.copy()
    for _ in range(32):
        if len(faces) > 1_000_000:
            raise ValueError("Surface refinement exceeds one million triangles")
        face_edges = faces[:, [[0, 1], [1, 2], [2, 0]]]
        edges, inverse = np.unique(np.sort(face_edges.reshape(-1, 2), axis=1), axis=0, return_inverse=True)
        split = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1) > max_edge_m * (1 + 1e-12)
        if not split.any():
            return vertices, faces
        midpoint = np.full(len(edges), -1, dtype=np.int64)
        midpoint[split] = len(vertices) + np.arange(split.sum())
        vertices = np.concatenate([vertices, vertices[edges[split]].mean(axis=1)])
        mids = midpoint[inverse].reshape(-1, 3)
        refined = []
        for (a, b, c), (ab, bc, ca) in zip(faces, mids):
            count = int(ab >= 0) + int(bc >= 0) + int(ca >= 0)
            if count == 0:
                refined.append((a, b, c))
            elif count == 3:
                refined.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
            elif count == 1:
                if bc >= 0:
                    a, b, c, ab = b, c, a, bc
                elif ca >= 0:
                    a, b, c, ab = c, a, b, ca
                refined.extend(((a, ab, c), (ab, b, c)))
            else:
                # Rotate so AB and BC are split; AC is the unsplit edge.
                if ab < 0:
                    a, b, c, ab, bc = b, c, a, bc, ca
                elif bc < 0:
                    a, b, c, ab, bc = c, a, b, ca, ab
                refined.extend(((ab, b, bc), (a, ab, c), (ab, bc, c)))
        faces = np.asarray(refined, dtype=np.int64)
    raise ValueError("Surface refinement did not converge")


def contains_points(vertices, faces, points):
    """Odd/even ray test; oblique rays avoid the STL's axis-aligned triangle edges."""
    tri = vertices[faces]
    direction = np.array([1., 0.371390676, 0.529127483])
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    h = np.cross(np.broadcast_to(direction, e2.shape), e2)
    det = np.einsum('ij,ij->i', e1, h)
    valid = np.abs(det) > 1e-14
    tri, e1, e2, h, det = tri[valid], e1[valid], e2[valid], h[valid], det[valid]
    result = []
    for start in range(0, len(points), 128):
        displacement = points[start:start + 128, None, :] - tri[None, :, 0]
        u = np.einsum('bfi,fi->bf', displacement, h) / det
        q = np.cross(displacement, e1)
        v = np.einsum('i,bfi->bf', direction, q) / det
        t = np.einsum('fi,bfi->bf', e2, q) / det
        hits = (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-10)
        result.extend((hits.sum(axis=1) % 2).astype(bool))
    return np.asarray(result, dtype=bool)


def tet_volumes(vertices, tets):
    x = vertices[tets]
    return np.linalg.det(np.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]], axis=-1)) / 6


def surface_faces(tets):
    faces = tets[:, [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]].reshape(-1, 3)
    _, index, counts = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    return faces[index[counts == 1]]


def component_count(tets, vertex_count):
    parents = np.arange(vertex_count)

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for tet in tets:
        for i in tet[1:]:
            parents[root(int(i))] = root(int(tet[0]))
    return len({root(i) for i in range(vertex_count)})


@dataclass
class TetMesh:
    vertices: np.ndarray
    tets: np.ndarray
    metadata: dict

    @property
    def faces(self):
        return surface_faces(self.tets)

    @property
    def volume(self):
        return float(tet_volumes(self.vertices, self.tets).sum())

    def validate(self):
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3 or self.tets.ndim != 2 or self.tets.shape[1] != 4:
            raise ValueError("Expected vertices (N,3) and tetrahedra (M,4)")
        if not len(self.tets) or not np.isfinite(self.vertices).all():
            raise ValueError("Empty or non-finite tetrahedral mesh")
        if self.tets.min() < 0 or self.tets.max() >= len(self.vertices):
            raise ValueError("Tetrahedral vertex index out of bounds")
        if np.any(tet_volumes(self.vertices, self.tets) <= 1e-18):
            raise ValueError("Inverted or degenerate reference tetrahedra")
        if component_count(self.tets, len(self.vertices)) != 1:
            raise ValueError("Disconnected volume mesh; use a finer pitch or Gmsh meshing")
        return self

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, vertices=self.vertices, tets=self.tets,
                            metadata=json.dumps(self.metadata, ensure_ascii=False))

    @classmethod
    def load(cls, config):
        path = resolve_asset(config.tetra_mesh)
        if not path.exists():
            raise FileNotFoundError(f"Missing volume mesh {path}. Run scripts/prepare_mesh.py first.")
        with np.load(path, allow_pickle=False) as data:
            mesh = cls(data['vertices'].copy(), data['tets'].copy(), json.loads(str(data['metadata'])))
        digest = hashlib.sha256(resolve_asset(config.stl).read_bytes()).hexdigest()
        if mesh.metadata.get('source_sha256') != digest or mesh.metadata.get('scale_to_m') != config.scale_to_m:
            raise ValueError("Volume mesh does not match the configured STL or units; regenerate it")
        mesh.validate()
        pinned = mesh.vertices[:, 2] <= mesh.vertices[:, 2].min() + config.pin_height_m + 1e-8
        if not pinned.any() or pinned.all():
            raise ValueError("Pin height must select the base and leave free deformable vertices")
        return mesh


def voxel_tetrahedralize(vertices, faces, pitch):
    """Explicit approximation: retain cubes with centers inside the STL, preserving resolved holes."""
    origin = vertices.min(0)
    shape = np.ceil((vertices.max(0) - origin) / pitch).astype(int)
    if np.prod(shape) > 2_000_000:
        raise ValueError("Voxel grid exceeds two million cells; use a coarser pitch or Gmsh")
    cells = np.stack(np.meshgrid(*(np.arange(n) for n in shape), indexing='ij'), axis=-1).reshape(-1, 3)
    cells = cells[contains_points(vertices, faces, origin + (cells + 0.5) * pitch)]
    if not len(cells):
        raise ValueError("No interior voxels; reduce pitch")
    offsets = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    # Six consistently oriented tetrahedra around each cube's 0->7 diagonal.
    pattern = np.array([[0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7],
                        [0, 6, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7]])
    lattice, inverse = np.unique((cells[:, None] + offsets).reshape(-1, 3), axis=0, return_inverse=True)
    points = origin + lattice * pitch
    tets = inverse.reshape(-1, 8)[:, pattern].reshape(-1, 4)
    return points, tets
