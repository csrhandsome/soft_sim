"""A compact tabletop stage with PBR materials, softboxes and physical supports."""
from pathlib import Path
import numpy as np


def material(stage, name, color, roughness=.45, metallic=0., texture=None):
    from pxr import Sdf, UsdShade, Gf
    m=UsdShade.Material.Define(stage, '/World/Looks/'+name)
    s=UsdShade.Shader.Define(stage, str(m.GetPath())+'/Surface')
    s.CreateIdAttr('UsdPreviewSurface')
    s.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    s.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
    s.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metallic)
    m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), 'surface')
    if texture:
        uv=UsdShade.Shader.Define(stage,str(m.GetPath())+'/UV')
        uv.CreateIdAttr('UsdPrimvarReader_float2')
        uv.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
        t=UsdShade.Shader.Define(stage,str(m.GetPath())+'/Texture')
        t.CreateIdAttr('UsdUVTexture')
        t.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(str(Path(texture).resolve()))
        t.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
        t.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(),'result')
        s.GetInput('diffuseColor').ConnectToSource(t.ConnectableAPI(),'rgb')
    return m


def bind(prim, mat):
    from pxr import UsdShade
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def box(stage, name, center, size, mat, collision=False):
    from pxr import UsdGeom, UsdPhysics, Gf, PhysxSchema
    c=UsdGeom.Cube.Define(stage,'/World/Set/'+name.replace('-', 'neg'))
    c.CreateSizeAttr(1.)
    c.AddTranslateOp().Set(Gf.Vec3d(*center))
    c.AddScaleOp().Set(Gf.Vec3f(*size))
    bind(c.GetPrim(),mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        p=PhysxSchema.PhysxCollisionAPI.Apply(c.GetPrim())
        p.CreateContactOffsetAttr(.0005); p.CreateRestOffsetAttr(0.)
    return c


def wood_texture():
    from shared.dataset import write_png
    out=Path(__file__).resolve().parents[2]/'data/scene_assets/oak_grain.png'
    if not out.exists():
        out.parent.mkdir(parents=True,exist_ok=True)
        rng=np.random.default_rng(7)
        y,x=np.mgrid[:512,:512]/512
        wave=np.sin(x*210+3*np.sin(y*8)+np.sin(y*28))
        grain=.12*wave+.03*rng.normal(size=x.shape)+.03*np.sin(x*1700)
        base=np.array([.49,.32,.17])
        rgb=np.clip(base[None,None,:]+grain[...,None]*np.array([.5,.36,.2]),0,1)
        write_png(out,(rgb*255).astype(np.uint8))
    return out


def build_presentation(env):
    from pxr import Usd, UsdGeom, UsdLux, Gf
    stage=env.stage; preset=env.config.simulation.scene_preset
    oak=material(stage,'Oak',(.45,.28,.13),.48,texture=wood_texture())
    metal=material(stage,'AnodizedAluminum',(.20,.23,.27),.28,.8)
    dark=material(stage,'Graphite',(.035,.045,.055),.48,.15)
    ceramic=material(stage,'Ceramic',(.82,.85,.84),.28)
    wall=material(stage,'Wall',(.35,.42,.48) if preset=='studio' else ((.21,.27,.29) if preset=='workshop' else (.76,.72,.65)),.85)
    mat=material(stage,'Mat',(.035,.10,.12),.85)
    if preset=='kitchen': oak=material(stage,'Counter',(.62,.61,.56),.26)
    box(stage,'Desktop',(0,.09,.0775),(.85,.65,.045),oak,True)
    for i,(x,y) in enumerate([(-.34,-.16),(.34,-.16),(-.34,.34),(.34,.34)]):
        box(stage,'Leg'+str(i),(x,y,.025),(.025,.025,.055),metal)
    box(stage,'BackWall',(0,.48,.35),(1.8,.03,.70),wall)
    box(stage,'WorkMat',(0,.015,.101),(.36,.29,.002),mat)
    # Marks give scale without adding collision geometry beside the soft fingers.
    for i in range(-8,9):
        box(stage,'GridX'+str(i),(i*.02,.015,.1021),(.00035,.27,.0001),metal)
    for i in range(-6,7):
        box(stage,'GridY'+str(i),(0,i*.02+.015,.1022),(.34,.00035,.0001),metal)
    box(stage,'Rail',(0,0,.108),(.25,.055,.012),metal)
    env.mount_ops=[]
    for side,name in enumerate(('left','right')):
        root=UsdGeom.Xform.Define(stage,'/World/Mount_'+name)
        env.mount_ops.append(root.AddTransformOp())
        c=UsdGeom.Cube.Define(stage,str(root.GetPath())+'/Clamp')
        c.CreateSizeAttr(1.);c.AddTranslateOp().Set(Gf.Vec3d(0,0,-.005))
        c.AddScaleOp().Set(Gf.Vec3f(.046,.046,.010));bind(c.GetPrim(),metal)
        # TPU material responds to the new area lights; no change to elasticity.
        bind(stage.GetPrimAtPath('/World/Finger_'+name),material(stage,'TPU_'+name,
             (.10,.43,.48) if side==0 else (.78,.29,.075),.52))
    # Support the free object at its initial rotated lower bound.
    bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render']).ComputeWorldBound(env.object_root.GetPrim()).ComputeAlignedRange()
    bottom=float(bounds.GetMin()[2]); top=max(.104,bottom-.0005)
    box(stage,'ObjectPedestal',(env.episode.object_position_m[0],env.episode.object_position_m[1],(.102+top)/2),
        (.024,.024,top-.102),ceramic,True)
    # Desktop context, placed outside the experiment's collision volume.
    box(stage,'Instrument',( .24,.20,.132),(.13,.10,.06),dark)
    box(stage,'InstrumentFace',(.24,.148,.135),(.10,.002,.034),metal)
    box(stage,'Screen',(.223,.146,.137),(.046,.001,.022),material(stage,'Screen',(.025,.17,.21),.15))
    box(stage,'Notebook',(-.24,.16,.107),(.13,.17,.010),material(stage,'Paper',(.67,.63,.52),.9))
    if preset=='workshop':
        for i in range(3):
            box(stage,'Storage'+str(i),(-.27+i*.19,.36,.15),(.15,.12,.10),metal)
    elif preset=='kitchen':
        box(stage,'BackSplash',(0,.457,.23),(1.2,.008,.26),ceramic)
    # Retain a dim dome for fill, add three shaped photographic lights.
    stage.GetPrimAtPath('/World/Light').GetAttribute('inputs:intensity').Set(450.)
    for name,pos,target,power,color,width in [
        ('Key',(-.30,-.22,.65),(0,0,.17),3000.,(1.,.88,.72),.38),
        ('Fill',(.38,-.12,.42),(0,0,.18),1800.,(.75,.86,1.),.30),
        ('Rim',(0,.34,.55),(0,0,.18),2800.,(1.,.95,.86),.26)]:
        light=UsdLux.RectLight.Define(stage,'/World/Lighting/'+name)
        light.CreateIntensityAttr(power); light.CreateColorAttr(Gf.Vec3f(*color))
        light.CreateWidthAttr(width);light.CreateHeightAttr(width)
        light.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*pos),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse())
    # Decorate primitive test objects while preserving downloaded asset materials.
    if not env.episode.object_shape.startswith('usd:'):
        bind(stage.GetPrimAtPath('/World/Object/Shape'),material(stage,'TestObject',(.36,.075,.045),.32))
