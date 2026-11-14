"""Geometry-aware soft-finger grasp candidates + mplib collision-aware planning.

Plans in the new soft_grasp_tcp frame. Does not treat the factory palm as the
contact center. A plan is only a geometric hypothesis until the FEM lift test.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import mplib
from mplib import Pose
from mplib.collision_detection import fcl
from soft_sim.config import ROOT
from soft_sim.mesh import load_stl
from soft_sim.geometry import matrix_to_quat_wxyz
from soft_sim.execution import support_placement_translation


def section_radius(vertices, faces, height):
    tri = vertices[faces]
    intersections = []
    for i,j in ((0,1),(1,2),(2,0)):
        a,b = tri[:,i],tri[:,j]
        valid = ((a[:,2]-height)*(b[:,2]-height) <= 0) & (np.abs(b[:,2]-a[:,2]) > 1e-10)
        t = (height-a[valid,2])/(b[valid,2]-a[valid,2])
        intersections.extend(a[valid]+t[:,None]*(b[valid]-a[valid]))
    return float(np.asarray(intersections)[:,0].max())


def bvh(vertices, faces):
    model=fcl.BVHModel()
    model.begin_model(len(faces),len(vertices))
    model.add_sub_model(list(np.asarray(vertices,dtype=float)),list(np.asarray(faces,dtype=np.int32)))
    model.end_model()
    return model


def surface_contacts(mesh, tcp, closing_axis):
    """Opposed first hits on actual triangles, including concave silhouettes."""
    first,_,left_tri=mesh.ray.intersects_location([tcp-closing_axis*.2],[closing_axis],multiple_hits=False)
    last,_,right_tri=mesh.ray.intersects_location([tcp+closing_axis*.2],[-closing_axis],multiple_hits=False)
    if not len(first) or not len(last):
        return None
    contacts=np.stack([first[0],last[0]])
    normals=mesh.face_normals[[left_tri[0],right_tri[0]]].copy()
    if normals[0]@closing_axis>0: normals[0]*=-1
    if normals[1]@closing_axis<0: normals[1]*=-1
    return contacts,normals


def centered_closure(contacts, closing_axis, radius, offset, indent, minimum_separation):
    """Place the two surfaces around the measured contacts with equal preload."""
    center=np.asarray(contacts).mean(axis=0)
    width=float((contacts[1]-contacts[0])@closing_axis)
    closed=max(minimum_separation-2*offset, width+2*radius-2*offset-2*indent)
    return center,width,closed


def plan_scene(scene_dir, out, *, contact_height=None, friction=.7, indent=.003,
               max_attempts=24, lift_height=.10, place_on=None, release_gap=.002,
               sample_dt=.02):
    """Plan one grasp (plus optional placement) on an exported scene directory."""
    scene_dir = Path(scene_dir).resolve()
    out = Path(out).resolve()
    scene=json.loads((scene_dir/"scene.json").read_text())
    geometry=np.load(scene_dir/"object_geometry.npz")
    contact_mesh=trimesh.Trimesh(geometry["vertices"],geometry["faces"],process=True)
    contact_mesh.update_faces(contact_mesh.unique_faces())
    contact_mesh.remove_unreferenced_vertices()
    object_mesh=contact_mesh.convex_hull
    placement_delta = None
    if place_on:
        with np.load(place_on) as support:
            placement_delta = support_placement_translation(contact_mesh.vertices, support["vertices"], release_gap)
    center=object_mesh.bounds.mean(axis=0)
    if scene.get("robot_name") != "x5":
        raise SystemExit("Only the dual-X5 scene is supported; export it with robodojo_task_soft.py")
    installation = scene.get("finger_installation")
    if installation is None:
        raise SystemExit("Scene has no measured finger installation; re-export it on the measured mounts")
    scale=scene.get("finger_scale",1.)
    vertices,faces=load_stl(ROOT/scene["finger_mesh"],scene.get("finger_scale_to_m",.001))
    height=contact_height if contact_height is not None else .055*scale
    radius=section_radius(vertices,faces,height)
    mount,offset=scene["hand_mount_offset_m"],scene["adapter_offset_m"]
    from soft_sim.robots.x5.mount_geometry import contact_geometry, tool_meshes
    radius, offset, rest_gap = contact_geometry(scale, installation, height)
    mounted_vertices = [[mesh.vertices.copy() for mesh in tool_meshes(scale, side, installation).values()]
                        for side in (0, 1)]
    table_z=scene["table_z"]
    out.mkdir(parents=True,exist_ok=True)
    hand_mesh=trimesh.load(ROOT/"third_party/RoboDojo/Assets/Robots/x5/meshes/link6.STL",force="mesh")
    hand_vertices=np.asarray(hand_mesh.vertices)
    from soft_sim.robots.x5.soft_tool import TOOL_ROTATION, MOUNT_Z, make_x5_urdf, JOINT_NAMES
    tool_in_hand=TOOL_ROTATION
    candidates=[]
    contact_cache={}
    for yaw in np.arange(0,360,45):
        c=np.array([np.cos(np.deg2rad(yaw)),np.sin(np.deg2rad(yaw)),0.])
        for direction in (1,-1):
            horizontal=direction*np.array([c[1],-c[0],0.])
            for tilt in (0,30,45,60,70,75,78,80,82,82.5,82.7,83,84,85,86,88):
                a=np.sin(np.deg2rad(tilt))*horizontal+np.array([0.,0.,-np.cos(np.deg2rad(tilt))])
                r=np.column_stack([c,np.cross(a,c),a])
                hand_r=r@tool_in_hand.T
                for zoffset in (0.,.001,.002,.003,.006,.009):
                    tcp=center+[0,0,zoffset]
                    key=(float(yaw),zoffset)
                    if key not in contact_cache:
                        contact_cache[key]=surface_contacts(contact_mesh,tcp,c)
                    if contact_cache[key] is None:
                        continue
                    contacts,normals=contact_cache[key]
                    antipodal=min(-float(normals[0]@c),float(normals[1]@c))
                    if antipodal < 1./np.sqrt(1.+friction**2):
                        continue
                    minimum_separation = 2 * offset
                    tcp,width,closed=centered_closure(contacts,c,radius,offset,indent,minimum_separation)
                    opened=.088
                    if (width+2*radius-closed-2*offset)/2 < .00075 or not 0 < closed < opened:
                        continue
                    hand_pos=tcp-hand_r@np.array([mount+height,0.,MOUNT_Z])
                    minimum=float((hand_vertices@hand_r.T+hand_pos)[:,2].min())
                    for jaw in (opened, closed):
                        for side in (0, 1):
                            for points in mounted_vertices[side]:
                                local = points + [(-1 if side == 0 else 1) * jaw / 2, 0., -height]
                                minimum=min(minimum,float((local@r.T+tcp)[:,2].min()))
                    clearance=minimum-table_z
                    if clearance < .002:
                        continue
                    candidates.append(dict(tcp=tcp.tolist(),rotation=r.tolist(),opened=opened,closed=closed,
                        contact_source="source_triangle_mesh",contacts_world_m=contacts.tolist(),normals_world=normals.tolist(),
                        width=width,tilt_deg=tilt,yaw_deg=float(yaw),clearance_m=clearance,
                        antipodal=antipodal,score=float(antipodal*10-abs(zoffset)*200-tilt*.005)))
    candidates.sort(key=lambda c:c["score"],reverse=True)
    (out/"candidates.json").write_text(json.dumps(candidates,indent=2))
    print("GEOMETRIC_CANDIDATES",len(candidates),"contact_radius_m",radius,flush=True)
    if not candidates:
        raise RuntimeError("No table-clear antipodal candidates for the current geometry")
    urdf=make_x5_urdf(out,height,scale,installation)
    joint_names=JOINT_NAMES
    n_arm=len(joint_names)-2
    planner=mplib.Planner(urdf=urdf,srdf=out/"soft_x5.srdf",move_group="soft_grasp_tcp",
        user_joint_names=joint_names,joint_vel_limits=np.full(n_arm,.45),joint_acc_limits=np.full(n_arm,.7))
    # Keep robot, table, target, and attached payload in one BASE frame.
    # Convert every world target explicitly. This avoids attached-body pose
    # inconsistencies when a non-identity robot base pose is used in mplib.
    planner.set_base_pose(Pose([0,0,0],[1,0,0,0]))
    world_from_base=np.asarray(scene.get("world_from_robot_base",np.eye(4)),dtype=float)
    if "world_from_robot_base" not in scene:
        world_from_base[:3,3]=[0.,0.,table_z]
    base_translation=world_from_base[:3,3]
    base_rotation=world_from_base[:3,:3]
    def base_pose(position, quaternion):
        rotation=Rotation.from_quat(np.r_[quaternion[1:],quaternion[0]]).as_matrix()
        return Pose((np.asarray(position)-base_translation)@base_rotation,
                    matrix_to_quat_wxyz(base_rotation.T@rotation))
    target_vertices=(object_mesh.vertices-base_translation)@base_rotation
    table_center=scene.get("table_center_m",[.35,0,table_z-.025])
    planner.planning_world.add_object("table",fcl.CollisionObject(
        fcl.Box(scene.get("table_size_m",[1.05,.8,.05])),base_pose(table_center,[1,0,0,0])))
    planner.planning_world.add_object("target",fcl.CollisionObject(bvh(target_vertices,object_mesh.faces)))
    obstacles_file=scene_dir/"obstacle_geometry.npz"
    if obstacles_file.exists():
        with np.load(obstacles_file) as obstacles:
            for index in range(int(obstacles["count"])):
                vertices_base=(obstacles[f"vertices_{index}"]-base_translation)@base_rotation
                hull=trimesh.Trimesh(vertices_base,obstacles[f"faces_{index}"],process=False).convex_hull
                planner.planning_world.add_object(f"obstacle_{index}",fcl.CollisionObject(bvh(hull.vertices,hull.faces)))
    acm=planner.planning_world.get_allowed_collision_matrix()
    acm.set_entry("base_link","table",True)
    for side,parent in (("left","link8"),("right","link7")):
        for component in ("soft_"+side,"soft_"+side+"_adapter","soft_"+side+"_camera"):
            acm.set_entry(component,[parent,"soft_"+side,
                "soft_"+side+"_adapter","soft_"+side+"_camera"],True)
    def cartesian_segment(start_q, start_position, end_position, quaternion):
        path=[np.asarray(start_q).copy()]
        count=max(2,int(np.ceil(np.linalg.norm(np.asarray(end_position)-start_position)/.002)))
        for t in np.linspace(0,1,count+1)[1:]:
            position=(1-t)*np.asarray(start_position)+t*np.asarray(end_position)
            status, solution=planner.IK(base_pose(position,quaternion), path[-1], mask=[False]*n_arm+[True,True],
                n_init_qpos=12, return_closest=True, threshold=.0005)
            if status!="Success":
                return {"status":"Cartesian IK: "+status+" fraction="+str(round(float(t),3))}
            for fraction in np.linspace(0,1,max(3,int(np.max(np.abs(solution-path[-1]))/.015)+1))[1:]:
                sample=(1-fraction)*path[-1]+fraction*solution
                if planner.check_for_env_collision(sample) or planner.check_for_self_collision(sample):
                    return {"status":"Cartesian interpolation collision"}
            path.append(solution.copy())
        times,positions,velocities,accelerations,duration=planner.TOPP(np.asarray(path)[:,:n_arm],sample_dt)
        return dict(status="Success",position=positions,time=times,duration=duration)
    attempts=[]
    for index,candidate in enumerate(candidates[:max_attempts]):
        q=np.array(scene["initial_q"])
        q[-2:]=candidate["opened"]/2
        tcp=np.array(candidate["tcp"])
        r=np.array(candidate["rotation"])
        quat=matrix_to_quat_wxyz(r)
        pre=tcp-r[:,2]*.09
        initial_collision=planner.check_for_env_collision(q)+planner.check_for_self_collision(q)
        if initial_collision:
            print("INITIAL_COLLISION",[(x.link_name1,x.link_name2) for x in initial_collision],flush=True)
            raise RuntimeError("Initial custom tool configuration is in collision")
        approach=planner.plan_pose(base_pose(pre,quat),q,mask=[False]*n_arm+[True,True],time_step=sample_dt,planning_time=2.,verbose=False)
        if approach["status"]!="Success":
            attempts.append(dict(index=index,phase="pregrasp",status=approach["status"]))
            continue
        qpre=np.r_[approach["position"][-1],q[-2:]]
        descend=cartesian_segment(qpre,pre,tcp,quat)
        if descend["status"]!="Success":
            attempts.append(dict(index=index,phase="final_approach",status=descend["status"]))
            continue
        qclosed=np.r_[descend["position"][-1],np.repeat(candidate["closed"]/2,2)]
        # Only intentional object/soft-finger contacts are allowed for closure.
        for side in ("soft_left","soft_right"):
            acm.set_entry(side,"target",True)
        closed_collisions=[]
        qopen=np.r_[descend["position"][-1],np.repeat(candidate["opened"]/2,2)]
        for fraction in np.linspace(0,1,12):
            sample=qopen*(1-fraction)+qclosed*fraction
            closed_collisions=planner.check_for_env_collision(sample)+planner.check_for_self_collision(sample)
            if closed_collisions:
                break
        if closed_collisions:
            np.save(out / "closed_collision_q.npy",qclosed)
            print("CLOSED_COLLISION",index,[(x.link_name1,x.link_name2) for x in closed_collisions],flush=True)
            for side in ("soft_left","soft_right"):
                acm.set_entry(side,"target",False)
            attempts.append(dict(index=index,phase="closed_geometry",status="collision"))
            continue
        planner.remove_object("target")
        carried=(object_mesh.vertices-tcp)@r
        planner.update_attached_object(bvh(carried,object_mesh.faces),Pose(),name="payload")
        payload_attached = True
        acm.set_entry("payload",["soft_left","soft_right"],True)
        probe=cartesian_segment(qclosed,tcp,tcp+[0,0,.012],quat)
        if probe["status"]=="Success":
            qprobe=np.r_[probe["position"][-1],qclosed[-2:]]
            lift=planner.plan_pose(base_pose(tcp+[0,0,lift_height],quat),qprobe,mask=[False]*n_arm+[True,True],
                time_step=sample_dt,planning_time=3.)
            lift_rotation=r
            if lift["status"]!="Success":
                down=np.array([0.,0.,-1.])
                lift_rotation=np.column_stack([r[:,0],np.cross(down,r[:,0]),down])
                lift=planner.plan_pose(base_pose(tcp+[0,0,lift_height],matrix_to_quat_wxyz(lift_rotation)),qprobe,
                    mask=[False]*n_arm+[True,True],time_step=sample_dt,planning_time=4.)
        else:
            lift=probe
        if lift["status"]!="Success":
            print("LIFT_DIAG",index,"probe",probe["status"],flush=True)
            if probe["status"]=="Success":
                planner.IK(base_pose(tcp+[0,0,lift_height],matrix_to_quat_wxyz(lift_rotation)),qprobe,
                    mask=[False]*n_arm+[True,True],n_init_qpos=3,return_closest=True,verbose=True)
            planner.detach_object("payload",also_remove=True)
            planner.planning_world.add_object("target",fcl.CollisionObject(bvh(target_vertices,object_mesh.faces)))
            for side in ("soft_left","soft_right"):
                acm.set_entry(side,"target",False)
            attempts.append(dict(index=index,phase="lift",status=lift["status"]))
            continue
        trajectories={"pregrasp":approach["position"],"approach":descend["position"],"probe":probe["position"],"lift":lift["position"]}
        placement = None
        if placement_delta is not None:
            goal = tcp+placement_delta
            above = goal+[0, 0, .09]
            qlift = np.r_[lift["position"][-1], qclosed[-2:]]
            transfer = dict(status="Transfer changes orientation")
            if np.allclose(lift_rotation, r, atol=1e-6):
                transfer = cartesian_segment(qlift, tcp+[0, 0, lift_height], above, quat)
            if transfer["status"] != "Success":
                transfer = planner.plan_pose(base_pose(above, quat), qlift,
                    mask=[False]*n_arm+[True, True], time_step=sample_dt, planning_time=5.)
            place = dict(status="Transfer: "+transfer["status"])
            retreat, home = place, place
            if transfer["status"] == "Success":
                place = cartesian_segment(np.r_[transfer["position"][-1], qclosed[-2:]], above, goal, quat)
            if place["status"] == "Success":
                planner.detach_object("payload", also_remove=True)
                payload_attached = False
                placed_vertices = (object_mesh.vertices+placement_delta-base_translation)@base_rotation
                planner.planning_world.add_object("placed_target", fcl.CollisionObject(
                    bvh(placed_vertices, object_mesh.faces)))
                acm.set_entry("placed_target", ["soft_left", "soft_right"], True)
                qplace = np.r_[place["position"][-1], qclosed[-2:]]
                qrelease = qplace.copy()
                qrelease[-2:] = candidate["opened"]/2
                release_valid = all(not (planner.check_for_env_collision((1-u)*qplace+u*qrelease)
                    +planner.check_for_self_collision((1-u)*qplace+u*qrelease)) for u in np.linspace(0, 1, 16))
                # Once open, the placed block is an ordinary obstacle. A
                # retreat must not rely on permitted finger/payload contact.
                acm.set_entry("placed_target", ["soft_left", "soft_right"], False)
                retreat = cartesian_segment(qrelease, goal, goal-r[:, 2]*.09, quat) if release_valid else dict(status="Release collision")
                if retreat["status"] == "Success":
                    acm.set_entry("placed_target", ["soft_left", "soft_right"], False)
                    home = planner.plan_qpos([q], np.r_[retreat["position"][-1], qrelease[-2:]],
                        time_step=sample_dt, planning_time=5.)
            failed = next(((name, path["status"]) for name, path in
                           (("transfer", transfer), ("place", place), ("retreat", retreat), ("return_home", home))
                           if path["status"] != "Success"), None)
            if failed:
                attempts.append(dict(index=index, phase=failed[0], status=failed[1]))
                if payload_attached:
                    planner.detach_object("payload", also_remove=True)
                planner.remove_object("placed_target")
                planner.planning_world.add_object("target", fcl.CollisionObject(bvh(target_vertices,object_mesh.faces)))
                for side in ("soft_left", "soft_right"):
                    acm.set_entry(side, "target", False)
                continue
            trajectories.update(transfer=transfer["position"], place=place["position"],
                release=np.repeat(place["position"][-1: ], 2, axis=0),
                retreat=retreat["position"], return_home=home["position"])
            placement = dict(object_translation_m=placement_delta.tolist(), tcp_goal_m=goal.tolist(),
                release_gap_m=release_gap, support_geometry=str(Path(place_on).resolve()),
                final_object_position=(np.asarray(scene["initial_object_position"])+placement_delta).tolist())
        np.savez_compressed(out/"joints.npz",**trajectories)
        report=dict(status="planned",candidate=candidate,world_from_robot_base_translation_m=base_translation.tolist(),
            world_from_robot_base=world_from_base.tolist(),lift_height_m=lift_height,contact_height_m=height,
            finger_inner_radius_m=radius,hand_to_tcp_m=mount+height,initial_q=q.tolist(),
            scene=scene,trajectory_file="joints.npz",sample_dt_s=sample_dt,
            planner="mplib RRT pregrasp/lift + collision-checked Cartesian approach/probe; full shared STL + adapter/camera proxies + carried object",
            collision_checked=True,physical_grasp_validated=False,friction_coefficient=friction,probe_lift_m=.012,lift_goal_rotation=lift_rotation.tolist(),attempts=attempts)
        if placement is not None:
            report["placement"] = placement
            report["planner"] += "; collision-checked transfer/place/release/retreat/return-home"
        (out/"plan.json").write_text(json.dumps(report,indent=2))
        print("SOFT_GRASP_PLAN",json.dumps({k:v for k,v in report.items() if k not in ("scene","attempts")}),flush=True)
        return
    (out/"attempts.json").write_text(json.dumps(attempts,indent=2))
    raise RuntimeError("No collision-free pregrasp/approach/closed/lift combination: "+str(attempts))

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene",type=Path,required=True,help="Exported scene directory")
    parser.add_argument("--output",type=Path,required=True,help="Plan output directory")
    parser.add_argument("--contact-height",type=float,default=None,
                        help="Contact height in meters on the scaled finger; default 55 mm times scale")
    parser.add_argument("--friction",type=float,default=.7,help="Conservative friction coefficient for contact-normal screening")
    parser.add_argument("--indent",type=float,default=.003)
    parser.add_argument("--max-attempts",type=int,default=24)
    parser.add_argument("--lift-height",type=float,default=.10,
                        help="TCP lift in meters; use margin above a task's object-lift threshold")
    parser.add_argument("--place-on", type=Path,
                        help="World-space support geometry NPZ; extend pickup with placement and return home")
    parser.add_argument("--release-gap",type=float,default=.002)
    parser.add_argument("--sample-dt",type=float,default=.02,help="Planner sample period, seconds")
    args=parser.parse_args(argv)
    plan_scene(args.scene, args.output, contact_height=args.contact_height,
               friction=args.friction, indent=args.indent, max_attempts=args.max_attempts,
               lift_height=args.lift_height, place_on=args.place_on,
               release_gap=args.release_gap, sample_dt=args.sample_dt)


if __name__=="__main__":
    raise SystemExit(main())
