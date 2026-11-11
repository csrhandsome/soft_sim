"""Interactive tabletop viewer; physics runs on ASUS and UI streams to the client."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import json
import time
import numpy as np
from shared.config import ExperimentConfig
from shared.experiment import EpisodeSpec, gripper_state
from backends.isaac_sim.app_launcher import add_livestream_arguments, configure_visualization
from backends.isaac_sim.soft_gripper_env import IsaacSimGripperEnv

parser=argparse.ArgumentParser()
add_livestream_arguments(parser)
parser.add_argument('--gui',action='store_true')
parser.add_argument('--scene',choices=['studio','workshop','kitchen'],default='studio')
parser.add_argument('--mode',choices=['fixed','free'],default='free')
parser.add_argument('--objects',nargs='+')
parser.add_argument('--seed',type=int,default=10)
parser.add_argument('--loop',action='store_true')
args=parser.parse_args()
cfg=ExperimentConfig()
cfg.simulation.scene_preset=args.scene
cfg.simulation.timestep_s=.001
cfg.simulation.solver_iterations=160
cfg.simulation.render=False
cfg.simulation.settle_duration_s=.30
cfg.simulation.capture_every_steps=20
cfg.gripper.profile='panda'
cfg.gripper.open_jaw_width_m=.080
cfg.gripper.final_jaw_width_range_m=(.034,.038)
cfg.gripper.mount_tilt_range_deg=(0.,0.)
cfg.gripper.close_duration_s=3.0
cfg.gripper.hold_duration_s=.7
cfg.object.fixed=args.mode=='fixed'
cfg.object.rotation_rpy_range_deg=((-12.,12.),(-12.,12.),(-80.,80.))
configure_visualization(parser,args,cfg.simulation)
cfg.validate()
root=Path(__file__).resolve().parents[2]
objects=args.objects or [('usd:data/scene_assets/ycb/'+name+'.usd') for name in ('003_cracker_box','005_tomato_soup_can','011_banana') if (root/'data/scene_assets/ycb'/ (name+'.usd')).exists()]
if not objects: objects=['box','cylinder','sphere']
print('SHOWCASE_OBJECTS',json.dumps(objects),flush=True)
env=IsaacSimGripperEnv(cfg)
import omni.ui as ui
from pxr import Gf, UsdGeom
state={'next':0,'quit':False,'scene':args.scene,'fixed':cfg.object.fixed,'view':'overview','restart':False,'paused':False,'collision_mesh':False}
index=0
window=ui.Window('Tabletop experiment',width=340,height=350)
window.position_x=880
window.position_y=50
with window.frame:
    with ui.VStack(spacing=6):
        title=ui.Label('Loading tabletop...',height=26)
        rotation=ui.Label('',height=22)
        mode_label=ui.Label('',height=22)
        phase=ui.Label('',height=22)
        with ui.HStack(height=30):
            ui.Button('Previous',clicked_fn=lambda:state.update(next=-1))
            ui.Button('Next object',clicked_fn=lambda:state.update(next=1))
            ui.Button('New rotation',clicked_fn=lambda:state.update(restart=True))
        ui.Button('Pause / resume',height=26,clicked_fn=lambda:state.update(paused=not state['paused']))
        ui.Button('Surface / collision mesh',height=26,clicked_fn=lambda:state.update(collision_mesh=not state['collision_mesh']))
        with ui.HStack(height=30):
            ui.Button('Free interaction',clicked_fn=lambda:state.update(fixed=False,restart=True))
            ui.Button('Fixed squeeze',clicked_fn=lambda:state.update(fixed=True,restart=True))
        with ui.HStack(height=30):
            for preset in ('studio','workshop','kitchen'):
                ui.Button(preset.title(),clicked_fn=lambda value=preset:state.update(scene=value,restart=True))
        with ui.HStack(height=30):
            ui.Button('Overview',clicked_fn=lambda:state.update(view='overview'))
            ui.Button('Contact close-up',clicked_fn=lambda:state.update(view='contact'))
        ui.Label('Switch objects, change their starting rotation,\nor inspect the contact area up close.',word_wrap=True,height=40)

def camera_view():
    if hasattr(env,'surface_meshes') and env.surface_meshes:
        from backends.isaac_sim.render_surface import update_surfaces
        update_surfaces(env)
        for name,mesh in zip(('left','right'),env.surface_meshes):
            UsdGeom.Imageable(env.stage.GetPrimAtPath('/World/Finger_'+name)).GetPurposeAttr().Set('default' if state['collision_mesh'] else 'guide')
            mesh.GetVisibilityAttr().Set('invisible' if state['collision_mesh'] else 'inherited')
    target=Gf.Vec3d(0,0,.169)
    eye=target+ (Gf.Vec3d(.19,-.27,.14) if state['view']=='overview' else Gf.Vec3d(.13,-.17,.055))
    camera=UsdGeom.Camera.Get(env.stage,'/World/OverviewCamera')
    if camera:
        camera.GetOrderedXformOps()[0].Set(Gf.Matrix4d().SetLookAt(eye,target,Gf.Vec3d(0,0,1)).GetInverse())
        camera.GetFocalLengthAttr().Set(32.)

report=[]
try:
    while env.app.is_running():
        cfg.object.fixed=state['fixed'];cfg.simulation.scene_preset=state['scene']
        episode=EpisodeSpec.sample(cfg,args.seed+index)
        episode.object_shape=objects[index%len(objects)]
        env.episode=episode;env.state=gripper_state(cfg.gripper,episode,0.)
        env.build_scene(episode)
        from backends.isaac_sim.render_surface import create_surfaces
        create_surfaces(env)
        camera_view()
        name=episode.object_shape.rsplit('/',1)[-1].replace('.usdz','').replace('.usd','')
        title.text=f'{index%len(objects)+1}/{len(objects)}  {name}'
        rotation.text='Initial rotation: '+', '.join(f'{x:.1f}°' for x in episode.object_rpy_deg)
        mode_label.text='Free object — responds to contact' if not cfg.object.fixed else 'Fixed object — squeeze test'
        state['restart']=False;state['next']=0
        print('SHOWCASE_START',json.dumps({'index':index,'object':episode.object_shape,'rpy_deg':episode.object_rpy_deg,'fixed':cfg.object.fixed,'scene':state['scene']}),flush=True)
        dt=cfg.simulation.timestep_s
        duration=cfg.simulation.settle_duration_s+cfg.gripper.close_duration_s+cfg.gripper.hold_duration_s
        first_pose=None;last_pose=None;failed=False
        count=int(round(duration/dt))
        for i in range(count):
            if not env.app.is_running() or state['restart'] or state['next']: break
            while state['paused'] and env.app.is_running() and not state['next'] and not state['restart']:
                env.world.render(); camera_view(); time.sleep(.015)
            t=max(0.,i*dt-cfg.simulation.settle_duration_s)
            env.state=gripper_state(cfg.gripper,episode,t)
            env.set_gripper_state(env.state)
            env.world.step(render=False)
            if i%20==0:
                env.world.render();camera_view()
                points=env.get_vertices_world()
                if not np.isfinite(points).all():
                    phase.text='Unstable state: choose another object / wider gap'
                    print('SHOWCASE_UNSTABLE',name,t,flush=True);failed=True;break
                last_pose=env.get_object_pose().tolist()
                if first_pose is None: first_pose=last_pose
                if i%500==0:
                    from shared.geometry import transform_points
                    rest=np.stack([transform_points(env.mesh.vertices,*env.finger_pose(side)) for side in (0,1)])
                    sample={'object':name,'time_s':t,'pose':last_pose,'max_displacement_mm':float(np.linalg.norm(points-rest,axis=-1).max()*1000)}
                    print('SHOWCASE_SAMPLE',json.dumps(sample),flush=True)
                phase.text=('Settling' if i*dt<cfg.simulation.settle_duration_s else 'Closing' if t<cfg.gripper.close_duration_s else 'Holding')+f'  sim {t:.2f}s'
        if not state['next'] and not state['restart']:
            report.append({'object':episode.object_shape,'initial_rpy_deg':episode.object_rpy_deg,'fixed':cfg.object.fixed,'finite':not failed,'first_pose':first_pose,'last_pose':last_pose})
            (root/'data/logs/tabletop_showcase_report.json').write_text(json.dumps(report,indent=2))
            print('SHOWCASE_DONE',json.dumps(report[-1]),flush=True)
            phase.text='Ready — choose Next object or New rotation' if not failed else phase.text
            end=time.monotonic()+ (6. if args.loop else float('inf'))
            while env.app.is_running() and not state['next'] and not state['restart'] and time.monotonic()<end:
                # Preserve the last physical state while the user inspects it.
                env.world.render();camera_view()
                time.sleep(.015)
            if args.loop and not state['restart'] and not state['next']:state['next']=1
        if state['next']: index=(index+state['next'])%len(objects)
        elif state['restart']: args.seed+=len(objects)
finally:
    env.close()
