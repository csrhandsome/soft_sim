"""Embed the original STL surface in the FEM tetrahedra for smooth display."""
import numpy as np
from soft_sim.mesh import load_stl, refine_surface
from soft_sim.config import resolve_asset
from soft_sim.fem_reduction import bind_fixed_surface


def create_tpu_material(stage, path, config):
    """Author an opaque, nonmetallic TPU shader instead of displayColor defaults."""
    from pxr import Gf, Sdf, UsdShade
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + '/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*config.color_linear))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(config.roughness)
    # Roughness alone still leaves a broad specular lobe. The specular workflow
    # explicitly suppresses that lobe for the default matte appearance.
    shader.CreateInput('useSpecularWorkflow', Sdf.ValueTypeNames.Int).Set(1)
    shader.CreateInput('specularColor', Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(config.specular_reflectivity))
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.)
    shader.CreateInput('clearcoat', Sdf.ValueTypeNames.Float).Set(0.)
    shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(1.)
    shader.CreateOutput('surface', Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return material


def bind_surface(mesh, config, *, max_edge_m=0., fixed_support_tet=None):
    vertices, faces=load_stl(resolve_asset(config.stl),config.scale_to_m)
    vertices, faces = refine_surface(vertices, faces, max_edge_m)
    indices, weights = bind_points(mesh, vertices)
    if fixed_support_tet is not None:
        indices, weights = bind_fixed_surface(mesh, config, vertices, indices, weights, fixed_support_tet)
    return vertices, faces, indices, weights


def bind_points(mesh, vertices):
    """Bind points in the reference frame to cooked simulation tetrahedra."""
    from scipy.spatial import cKDTree
    tetra=mesh.vertices[mesh.tets]
    centers = tetra.mean(axis=1)
    tree = cKDTree(centers)
    lower, upper = tetra.min(axis=1), tetra.max(axis=1)
    maximum_radius = np.linalg.norm(tetra - centers[:, None], axis=-1).max() + 1e-9
    all_indices, all_weights = [], []
    # Refined display meshes can have tens of thousands of vertices. Batch
    # the candidate inversions rather than allocating N * 96 matrices at once.
    for start in range(0, len(vertices), 512):
        batch = vertices[start:start + 512]
        _, candidates = tree.query(batch, k=list(range(1, min(96, len(tetra)) + 1)))
        candidate_points = tetra[candidates]
        dm = np.stack([candidate_points[:, :, i] - candidate_points[:, :, 0] for i in (1, 2, 3)], axis=-1)
        tail = np.einsum('nkij,nkj->nki', np.linalg.inv(dm), batch[:, None, :] - candidate_points[:, :, 0])
        weights = np.concatenate([(1-tail.sum(axis=-1))[..., None], tail], axis=-1)
        penalty = np.maximum(-weights, 0).sum(axis=-1) + np.maximum(weights-1, 0).sum(axis=-1)
        chosen = penalty.argmin(axis=1)
        selected_tets = candidates[np.arange(len(batch)), chosen].copy()
        selected_weights = weights[np.arange(len(batch)), chosen].copy()
        # A containing long tet need not have one of the 96 nearest centroids,
        # especially beside finely meshed fillets. A radius/AABB fallback
        # considers every possible containing tet before extrapolating.
        for i in np.flatnonzero(penalty[np.arange(len(batch)), chosen] > 1e-6):
            point = batch[i]
            possible = np.asarray(tree.query_ball_point(point, maximum_radius), dtype=np.int64)
            possible = possible[np.all((point >= lower[possible] - 1e-9)
                                       & (point <= upper[possible] + 1e-9), axis=1)]
            if not len(possible):
                continue
            cells = tetra[possible]
            dm = np.stack([cells[:, j] - cells[:, 0] for j in (1, 2, 3)], axis=-1)
            tail = np.linalg.solve(dm, (point - cells[:, 0])[..., None])[..., 0]
            extra = np.c_[1 - tail.sum(axis=1), tail]
            cost = np.maximum(-extra, 0).sum(axis=1) + np.maximum(extra - 1, 0).sum(axis=1)
            best = cost.argmin()
            if cost[best] < penalty[i, chosen[i]]:
                selected_tets[i], selected_weights[i] = possible[best], extra[best]
        all_indices.append(mesh.tets[selected_tets])
        all_weights.append(selected_weights)
    indices, weights = np.concatenate(all_indices), np.concatenate(all_weights)
    reconstructed=np.einsum('ni,nij->nj',weights,mesh.vertices[indices])
    if not np.allclose(reconstructed,vertices,atol=1e-8):
        raise RuntimeError('Invalid FEM surface embedding')
    return indices,weights


def normals(vertices,faces):
    n=np.zeros_like(vertices)
    triangles=vertices[faces]
    face_normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    for corner in range(3):np.add.at(n,faces[:,corner],face_normals)
    return (n/np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-15)).astype(np.float32)


def create_surfaces(env):
    from pxr import UsdGeom, Vt, UsdShade
    if not hasattr(env,'_surface_binding'):
        env._surface_binding=bind_surface(env.mesh,env.config.mesh)
    _,faces,_,_=env._surface_binding
    env.surface_meshes=[]
    for name in ('left','right'):
        mesh=UsdGeom.Mesh.Define(env.stage,'/World/FingerSurface_'+name)
        mesh.CreateFaceVertexCountsAttr([3]*len(faces))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.astype(np.int32).ravel()))
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.SetNormalsInterpolation('vertex')
        env.surface_meshes.append(mesh)
        UsdGeom.Imageable(env.stage.GetPrimAtPath('/World/Finger_'+name)).CreatePurposeAttr('guide')
        material=UsdShade.Material.Get(env.stage,'/World/Looks/TPU_'+name)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    update_surfaces(env)


def update_surfaces(env):
    from pxr import Vt
    _,faces,indices,weights=env._surface_binding
    for view,mesh in zip(env.views,env.surface_meshes):
        nodes=view.get_nodal_positions()[0].numpy()[0]
        points=np.einsum('ni,nij->nj',weights,nodes[indices]).astype(np.float32)
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
        mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals(points,faces)))
