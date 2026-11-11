"""Geometry-aware soft-finger grasp candidates + mplib collision-aware planning.

Plans in the new soft_grasp_tcp frame. Does not treat the factory palm as the
contact center. A plan is only a geometric hypothesis until the FEM lift test.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import mplib
from mplib import Pose
from mplib.collision_detection import fcl
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.mesh import load_stl
from shared.geometry import matrix_to_quat_wxyz


def section_radius(vertices, faces, height):
    tri = vertices[faces]
    intersections = []
    for i,j in ((0,1),(1,2),(2,0)):
        a,b = tri[:,i],tri[:,j]
        valid = ((a[:,2]-height)*(b[:,2]-height) <= 0) & (np.abs(b[:,2]-a[:,2]) > 1e-10)
        t = (height-a[valid,2])/(b[valid,2]-a[valid,2])
        intersections.extend(a[valid]+t[:,None]*(b[valid]-a[valid]))
    return float(np.asarray(intersections)[:,0].max())


def make_urdf(out, height, mount, offset):
    source = ROOT / "RoboTwin/assets/embodiments/franka-panda"
    tree = ET.parse(source / "panda.urdf")
    robot = tree.getroot()
    for link in robot.findall("link"):
        for mesh in link.findall(".//mesh"):
            mesh.set("filename", os.path.relpath(source / mesh.attrib["filename"],out))
        if link.attrib["name"] in ("panda_leftfinger","panda_rightfinger"):
            for collision in list(link.findall("collision")):
                link.remove(collision)
    def fixed(name,parent,xyz,rpy=(0,0,0),mesh=None,box=None):
        link = ET.SubElement(robot,"link",name=name)
        if mesh or box:
            collision = ET.SubElement(link,"collision")
            geometry = ET.SubElement(collision,"geometry")
            if mesh:
                ET.SubElement(geometry,"mesh",filename=os.path.relpath(mesh,out),scale="0.001 0.001 0.001")
            else:
                # Keep fixtures as triangle meshes, matching the STL path.
                # This also avoids mixed primitive/BVH broadphase inconsistencies
                # in mplib 0.2.1 at millimetre-scale jaw clearances.
                mesh_file=out/(name+".obj")
                trimesh.creation.box(extents=box).export(mesh_file)
                ET.SubElement(geometry,"mesh",filename=mesh_file.name)
        joint = ET.SubElement(robot,"joint",name=name+"_joint",type="fixed")
        ET.SubElement(joint,"parent",link=parent)
        ET.SubElement(joint,"child",link=name)
        ET.SubElement(joint,"origin",xyz=" ".join(map(str,xyz)),rpy=" ".join(map(str,rpy)))
    for side,parent,sign,yaw in (("left","panda_rightfinger",-1,np.pi/2),
                                 ("right","panda_leftfinger",1,3*np.pi/2)):
        name = "soft_"+side
        fixed(name,parent,[0,sign*offset,mount-.0584],(0,0,yaw),mesh=ROOT/"data/assetes/soft_finger.stl")
        fixed(name+"_adapter",name,[0,0,-.0083],box=(.04,.04,.0166))
        # Camera lies inside the shared finger base. Housing dimensions here
        # are an explicit 12x12x8 mm simulation proxy, pending CAD calibration.
        fixed(name+"_camera",name,[0,0,.006],box=(.012,.012,.008))
    fixed("soft_grasp_tcp","panda_hand",[0,0,mount+height],(0,0,np.pi/2))
    path = out / "soft_franka.urdf"
    tree.write(path,encoding="utf-8",xml_declaration=True)
    shutil.copyfile(source/"panda.srdf",out/"soft_franka.srdf")
    return path


def bvh(vertices, faces):
    model=fcl.BVHModel()
    model.begin_model(len(faces),len(vertices))
    model.add_sub_model(list(np.asarray(vertices,dtype=float)),list(np.asarray(faces,dtype=np.int32)))
    model.end_model()
    return model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene",default="data/generated/soft_grasp_scene")
    parser.add_argument("--output",default="data/generated/soft_grasp_plan")
    parser.add_argument("--contact-height",type=float,default=.055)
    parser.add_argument("--friction",type=float,default=.7,help="Conservative friction coefficient for contact-normal screening")
    parser.add_argument("--indent",type=float,default=.003)
    parser.add_argument("--max-attempts",type=int,default=24)
    args=parser.parse_args()
    scene_dir=ROOT/args.scene
    scene=json.loads((scene_dir/"scene.json").read_text())
    geometry=np.load(scene_dir/"object_geometry.npz")
    object_mesh=trimesh.Trimesh(geometry["vertices"],geometry["faces"],process=True).convex_hull
    center=object_mesh.bounds.mean(axis=0)
    vertices,faces=load_stl(ROOT/scene["finger_mesh"])
    height=args.contact_height
    radius=section_radius(vertices,faces,height)
    mount,offset=scene["hand_mount_offset_m"],scene["adapter_offset_m"]
    table_z=scene["table_z"]
    out=ROOT/args.output
    out.mkdir(parents=True,exist_ok=True)
    hand_mesh=trimesh.load(ROOT/"RoboTwin/assets/embodiments/franka-panda/franka_description/meshes/collision/hand.stl",force="mesh")
    hand_vertices=np.asarray(hand_mesh.vertices)
    tool_in_hand=Rotation.from_euler("z",90,degrees=True).as_matrix()
    candidates=[]
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
                    first,_,left_tri=object_mesh.ray.intersects_location([tcp-c*.2],[c],multiple_hits=False)
                    last,_,right_tri=object_mesh.ray.intersects_location([tcp+c*.2],[-c],multiple_hits=False)
                    if not len(first) or not len(last):
                        continue
                    contacts=np.stack([first[0],last[0]])
                    normals=object_mesh.face_normals[[left_tri[0],right_tri[0]]]
                    antipodal=min(-float(normals[0]@c),float(normals[1]@c))
                    if antipodal < 1./np.sqrt(1.+args.friction**2):
                        continue
                    width=float((contacts[1]-contacts[0])@c)
                    closed=max(.043-2*offset, width+2*radius-2*offset-2*args.indent)
                    opened=.08  # full opening clears the whole tapered object, beyond the contact slice
                    if (width+2*radius-closed-2*offset)/2 < .00075 or not 0 < closed < opened <= .08:
                        continue
                    hand_pos=tcp-a*(mount+height)
                    minimum=float((hand_vertices@hand_r.T+hand_pos)[:,2].min())
                    for jaw in (opened,closed):
                        for sign in (-1,1):
                            rotation=np.eye(3) if sign==-1 else np.diag([-1.,-1.,1.])
                            local=vertices@rotation.T+[sign*(jaw/2+offset),0.,-height]
                            minimum=min(minimum,float((local@r.T+tcp)[:,2].min()))
                            corners=np.array([[x,y,z] for x in (-.020,.020) for y in (-.020,.020) for z in (-.0166,0.)])
                            corners += [sign*(jaw/2+offset),0.,-height]
                            minimum=min(minimum,float((corners@r.T+tcp)[:,2].min()))
                    clearance=minimum-table_z
                    if clearance < .002:
                        continue
                    candidates.append(dict(tcp=tcp.tolist(),rotation=r.tolist(),opened=opened,closed=closed,
                        width=width,tilt_deg=tilt,yaw_deg=float(yaw),clearance_m=clearance,
                        antipodal=antipodal,score=float(antipodal*10-abs(zoffset)*200-tilt*.005)))
    candidates.sort(key=lambda c:c["score"],reverse=True)
    (out/"candidates.json").write_text(json.dumps(candidates,indent=2))
    print("GEOMETRIC_CANDIDATES",len(candidates),"contact_radius_m",radius,flush=True)
    if not candidates:
        raise RuntimeError("No table-clear antipodal candidates for the current geometry")
    urdf=make_urdf(out,height,mount,offset)
    joint_names=["panda_joint"+str(i) for i in range(1,8)]+["panda_finger_joint1","panda_finger_joint2"]
    planner=mplib.Planner(urdf=urdf,srdf=out/"soft_franka.srdf",move_group="soft_grasp_tcp",
        user_joint_names=joint_names,joint_vel_limits=np.full(7,.45),joint_acc_limits=np.full(7,.7))
    # Keep robot, table, target, and attached payload in one BASE frame.
    # Convert every world target explicitly. This avoids attached-body pose
    # inconsistencies when a non-identity robot base pose is used in mplib.
    planner.set_base_pose(Pose([0,0,0],[1,0,0,0]))
    base_translation=np.array([0.,0.,table_z])
    target_vertices=object_mesh.vertices-base_translation
    planner.planning_world.add_object("table",fcl.CollisionObject(fcl.Box([1.05,.8,.05]),Pose([.35,0,-.025],[1,0,0,0])))
    planner.planning_world.add_object("target",fcl.CollisionObject(bvh(target_vertices,object_mesh.faces)))
    acm=planner.planning_world.get_allowed_collision_matrix()
    acm.set_entry("panda_link0","table",True)
    for side,parent in (("left","panda_rightfinger"),("right","panda_leftfinger")):
        for component in ("soft_"+side,"soft_"+side+"_adapter","soft_"+side+"_camera"):
            acm.set_entry(component,["panda_hand","panda_link7",parent,"soft_"+side,
                "soft_"+side+"_adapter","soft_"+side+"_camera"],True)
    def cartesian_segment(start_q, start_position, end_position, quaternion):
        path=[np.asarray(start_q).copy()]
        count=max(2,int(np.ceil(np.linalg.norm(np.asarray(end_position)-start_position)/.002)))
        for t in np.linspace(0,1,count+1)[1:]:
            position=(1-t)*np.asarray(start_position)+t*np.asarray(end_position)
            status, solution=planner.IK(Pose(position-[0,0,table_z],quaternion), path[-1], mask=[False]*7+[True,True],
                n_init_qpos=12, return_closest=True, threshold=.0005)
            if status!="Success":
                return {"status":"Cartesian IK: "+status+" fraction="+str(round(float(t),3))}
            for fraction in np.linspace(0,1,max(3,int(np.max(np.abs(solution-path[-1]))/.015)+1))[1:]:
                sample=(1-fraction)*path[-1]+fraction*solution
                if planner.check_for_env_collision(sample) or planner.check_for_self_collision(sample):
                    return {"status":"Cartesian interpolation collision"}
            path.append(solution.copy())
        times,positions,velocities,accelerations,duration=planner.TOPP(np.asarray(path)[:,:7],.02)
        return dict(status="Success",position=positions,time=times,duration=duration)
    attempts=[]
    for index,candidate in enumerate(candidates[:args.max_attempts]):
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
        approach=planner.plan_pose(Pose(pre-base_translation,quat),q,mask=[False]*7+[True,True],time_step=.02,planning_time=2.,verbose=False)
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
        closed_collisions=planner.check_for_env_collision(qclosed)+planner.check_for_self_collision(qclosed)
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
        acm.set_entry("payload",["soft_left","soft_right"],True)
        probe=cartesian_segment(qclosed,tcp,tcp+[0,0,.012],quat)
        if probe["status"]=="Success":
            qprobe=np.r_[probe["position"][-1],qclosed[-2:]]
            lift=planner.plan_pose(Pose(tcp+[0,0,.10]-base_translation,quat),qprobe,mask=[False]*7+[True,True],
                time_step=.02,planning_time=3.)
            lift_rotation=r
            if lift["status"]!="Success":
                down=np.array([0.,0.,-1.])
                lift_rotation=np.column_stack([r[:,0],np.cross(down,r[:,0]),down])
                lift=planner.plan_pose(Pose(tcp+[0,0,.10]-base_translation,matrix_to_quat_wxyz(lift_rotation)),qprobe,
                    mask=[False]*7+[True,True],time_step=.02,planning_time=4.)
        else:
            lift=probe
        if lift["status"]!="Success":
            print("LIFT_DIAG",index,"probe",probe["status"],flush=True)
            if probe["status"]=="Success":
                planner.IK(Pose(tcp+[0,0,.10-table_z],matrix_to_quat_wxyz(lift_rotation)),qprobe,
                    mask=[False]*7+[True,True],n_init_qpos=3,return_closest=True,verbose=True)
            planner.detach_object("payload",also_remove=True)
            planner.planning_world.add_object("target",fcl.CollisionObject(bvh(target_vertices,object_mesh.faces)))
            for side in ("soft_left","soft_right"):
                acm.set_entry(side,"target",False)
            attempts.append(dict(index=index,phase="lift",status=lift["status"]))
            continue
        trajectories={"pregrasp":approach["position"],"approach":descend["position"],"probe":probe["position"],"lift":lift["position"]}
        np.savez_compressed(out/"joints.npz",**trajectories)
        report=dict(status="planned",candidate=candidate,world_from_robot_base_translation_m=base_translation.tolist(),contact_height_m=height,
            finger_inner_radius_m=radius,hand_to_tcp_m=mount+height,initial_q=q.tolist(),
            scene=scene,trajectory_file="joints.npz",sample_dt_s=.02,
            planner="mplib RRT pregrasp/lift + collision-checked Cartesian approach/probe; full shared STL + adapter/camera proxies + carried object",
            collision_checked=True,physical_grasp_validated=False,friction_coefficient=args.friction,probe_lift_m=.012,lift_goal_rotation=lift_rotation.tolist(),attempts=attempts)
        (out/"plan.json").write_text(json.dumps(report,indent=2))
        print("SOFT_GRASP_PLAN",json.dumps({k:v for k,v in report.items() if k not in ("scene","attempts")}),flush=True)
        return
    (out/"attempts.json").write_text(json.dumps(attempts,indent=2))
    raise RuntimeError("No collision-free pregrasp/approach/closed/lift combination: "+str(attempts))

if __name__=="__main__":
    main()
