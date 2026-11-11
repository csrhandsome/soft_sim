"""Embed the original STL surface in the FEM tetrahedra for smooth display."""
import numpy as np
from shared.mesh import load_stl
from shared.config import resolve_asset


def bind_surface(mesh, config):
    from scipy.spatial import cKDTree
    vertices, faces=load_stl(resolve_asset(config.stl),config.scale_to_m)
    tetra=mesh.vertices[mesh.tets]
    _, candidates=cKDTree(tetra.mean(axis=1)).query(vertices,k=min(96,len(tetra)))
    candidate_points=tetra[candidates]
    dm=np.stack([candidate_points[:,:,i]-candidate_points[:,:,0] for i in (1,2,3)],axis=-1)
    tail=np.einsum('nkij,nkj->nki',np.linalg.inv(dm),vertices[:,None,:]-candidate_points[:,:,0])
    weights=np.concatenate([(1-tail.sum(axis=-1))[...,None],tail],axis=-1)
    penalty=np.maximum(-weights,0).sum(axis=-1)+np.maximum(weights-1,0).sum(axis=-1)
    chosen=penalty.argmin(axis=1)
    indices=mesh.tets[candidates[np.arange(len(vertices)),chosen]]
    weights=weights[np.arange(len(vertices)),chosen]
    reconstructed=np.einsum('ni,nij->nj',weights,mesh.vertices[indices])
    if not np.allclose(reconstructed,vertices,atol=1e-8):
        raise RuntimeError('Invalid FEM surface embedding')
    return vertices,faces,indices,weights


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
