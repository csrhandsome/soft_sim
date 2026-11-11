"""GPU diagnostic: separate inertial/gravity deformation from object contact."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import numpy as np
from shared.config import ExperimentConfig
from shared.experiment import EpisodeSpec, gripper_state
from shared.geometry import transform_points
from backends.isaac_sim.soft_gripper_env import IsaacSimGripperEnv

cfg = ExperimentConfig.load('configs/isaac_webrtc_preview.json')
cfg.simulation.render = False
cfg.simulation.livestream = False
cfg.simulation.scene_preset = 'plain'
cfg.object.shapes = ('sphere',)
cfg.object.radius_range_m = (.02, .02)
cfg.gripper.mount_tilt_range_deg = (0., 0.)
cfg.gripper.final_jaw_width_range_m = (.04, .04)
env = IsaacSimGripperEnv(cfg)
results = []
try:
    for label, gravity, moving, collision, offset in [
        ('stationary_zero_g', (0.,0.,0.), False, False, .0005),
        ('stationary_gravity', (0.,0.,-9.81), False, False, .0005),
        ('moving_zero_g', (0.,0.,0.), True, False, .0005),
        ('moving_no_object_contact', (0.,0.,-9.81), True, False, .0005),
        ('moving_contact_0p5mm', (0.,0.,-9.81), True, True, .0005),
    ]:
        cfg.simulation.gravity_m_s2 = gravity
        cfg.simulation.contact_offset_m = offset
        cfg.object.collision = collision
        episode = EpisodeSpec.sample(cfg, 0)
        env.episode = episode
        env.state = gripper_state(cfg.gripper, episode, 0.)
        env.build_scene(episode)
        samples=[]
        for i in range(2201):
            t=i*cfg.simulation.timestep_s
            env.state=gripper_state(cfg.gripper,episode,t if moving else 0.)
            env.set_gripper_state(env.state)
            env.world.step(render=False)
            if i % 200 == 0:
                p=env.get_vertices_world()
                rest=np.stack([transform_points(env.mesh.vertices,*env.finger_pose(side)) for side in (0,1)])
                finite=bool(np.isfinite(p).all())
                samples.append({'time_s':t,'finite':finite,'max_displacement_mm':float(np.linalg.norm(p-rest,axis=-1).max()*1000) if finite else None})
                if not finite: break
        results.append({'case':label,'samples':samples})
        Path('data/logs/contact_diagnostic.json').write_text(json.dumps(results,indent=2))
        print('DIAGNOSTIC',label,json.dumps(samples),flush=True)
    from isaacsim.storage.native import get_assets_root_path
    import omni.client
    from pxr import Sdf, UsdUtils
    root=get_assets_root_path()
    print('ASSET_ROOT',root,flush=True)
    out=Path('data/scene_assets');out.mkdir(exist_ok=True)
    for name, fragment in [('cracker_box','/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd'),('soup_can','/Isaac/Props/YCB/Axis_Aligned/005_tomato_soup_can.usd'),('banana','/Isaac/Props/YCB/Axis_Aligned/011_banana.usd')]:
        uri=root+fragment
        status,entry=omni.client.stat(uri)
        print('ASSET_STAT',name,str(status),flush=True)
        if status == omni.client.Result.OK:
            try:
                okay=UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(uri),str(out/(name+'.usdz')))
                print('ASSET_CACHED',name,okay,flush=True)
            except Exception as exc: print('ASSET_ERROR',name,repr(exc),flush=True)
except Exception:
    import traceback;traceback.print_exc()
finally:
    env.close()
