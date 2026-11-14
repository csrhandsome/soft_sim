"""Compare the serialized planning URDF with the shared simulation mount math."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import mplib
from scipy.spatial.transform import Rotation
from soft_sim.config import ROOT
from soft_sim.planning.grasp import section_radius, surface_contacts, centered_closure
from soft_sim.config import ExperimentConfig
from soft_sim.mesh import load_stl
from soft_sim.robots.x5.soft_tool import make_x5_urdf, mounts_from_q, JOINT_NAMES, MOUNT_X, MOUNT_Z, TOOL_ROTATION, scale_config
from soft_sim.mesh import TetMesh
from soft_sim.robots.x5.mount_geometry import measured_installation, contact_geometry, tool_meshes, fixture_meshes, validate_installation, camera_housing_surface
from soft_sim.geometry import mounted_camera_pose

class FrameIntegration(unittest.TestCase):
    def test_stl_contact_band_has_correct_units(self):
        vertices,faces=load_stl(ROOT/"assets/meshes/soft_finger.stl")
        self.assertTrue(.088 < vertices[:,2].max() < .089)
        radius=section_radius(vertices,faces,.055)
        self.assertTrue(.009 < radius < .011)
        self.assertGreater(radius,section_radius(vertices,faces,.073))

    def test_concave_target_contacts_do_not_grasp_empty_hull_space(self):
        import trimesh
        parts=[]
        for dimensions,center in (((.04,.01,.04),(0.,-.03,0.)),
                                  ((.04,.01,.04),(0.,.03,0.)),
                                  ((.005,.06,.04),(.0175,0.,0.))):
            box=trimesh.creation.box(extents=dimensions)
            box.apply_translation(center)
            parts.append(box)
        target=trimesh.util.concatenate(parts)
        axis=np.array([1.,0.,0.])
        actual,normals=surface_contacts(target,np.zeros(3),axis)
        hull,_=surface_contacts(target.convex_hull,np.zeros(3),axis)
        self.assertAlmostEqual(float((actual[1]-actual[0])@axis),.005)
        self.assertAlmostEqual(float((hull[1]-hull[0])@axis),.04)
        np.testing.assert_allclose(normals,[[-1,0,0],[1,0,0]])

    def test_asymmetric_surface_hits_get_equal_left_and_right_preload(self):
        contacts=np.array([[-.010,0.,.1],[.040,0.,.1]])
        axis=np.array([1.,0.,0.])
        radius,offset,indent=.005,.015,.003
        center,width,closed=centered_closure(contacts,axis,radius,offset,indent,.025)
        left_surface=center-axis*(closed/2+offset-radius)
        right_surface=center+axis*(closed/2+offset-radius)
        self.assertAlmostEqual(float((left_surface-contacts[0])@axis),indent)
        self.assertAlmostEqual(float((contacts[1]-right_surface)@axis),indent)
        self.assertAlmostEqual(width,.05)

    @unittest.skipUnless((ROOT/"third_party/RoboDojo/Assets/Robots/x5/X5A.urdf").exists(), "Download native X5 assets")
    def test_native_x5_planner_matches_mounts_with_asymmetric_slides(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            for scale, installation in [(s, measured_installation(s)) for s in (.65,.75,.8)]:
                height=.055*scale
                urdf=make_x5_urdf(out,height,scale,installation)
                planner=mplib.Planner(urdf=urdf,srdf=out/"soft_x5.srdf",move_group="soft_grasp_tcp",user_joint_names=JOINT_NAMES)
                base=np.eye(4)
                base[:3,:3]=Rotation.from_euler("xyz",[.1,-.2,1.57]).as_matrix()
                base[:3,3]=[.3,-.45,.765]
                for slides in ((.044,.044),(.012,.019),(0.,0.)):
                    q=np.r_[[.3,-.4,.5,-.7,.8,-.6],slides]
                    kin=planner.pinocchio_model
                    kin.compute_forward_kinematics(q)
                    expected=mounts_from_q(q,base,installation)
                    for side,name in enumerate(("soft_left","soft_right")):
                        actual=base@kin.get_link_pose(planner.link_name_2_idx[name]).to_transformation_matrix()
                        np.testing.assert_allclose(actual,expected[side],atol=2e-7)
                        camera = ExperimentConfig().camera
                        np.testing.assert_allclose(mounted_camera_pose(camera, actual[:3,3], actual[:3,:3]),
                            mounted_camera_pose(camera, expected[side,:3,3], expected[side,:3,:3]), atol=2e-7)
                        # Fixtures serialize slide-frame vertices. Compare
                        # every world vertex with the independent FK frame.
                        from soft_sim.robots.x5.soft_tool import link_transforms, FINGER_LINKS
                        import trimesh
                        for component, mesh in fixture_meshes(scale, side, installation).items():
                            fixture = base@kin.get_link_pose(planner.link_name_2_idx[name+"_"+component]).to_transformation_matrix()
                            saved = trimesh.load(out/(name+"_"+component+".obj"),force="mesh",process=False)
                            expected_pose = base@link_transforms(q)[FINGER_LINKS[side]]
                            np.testing.assert_allclose(saved.vertices@fixture[:3,:3].T+fixture[:3,3],
                                mesh.vertices@expected_pose[:3,:3].T+expected_pose[:3,3],atol=2e-7)
                    wrist=base@kin.get_link_pose(planner.link_name_2_idx["link6"]).to_transformation_matrix()
                    tcp=base@kin.get_link_pose(planner.link_name_2_idx["soft_grasp_tcp"]).to_transformation_matrix()
                    forward = installation["forward_offset_m"]
                    np.testing.assert_allclose(tcp[:3,3],(wrist@np.array([MOUNT_X+height+forward,0.,MOUNT_Z,1.]))[:3],atol=2e-7)
                    np.testing.assert_allclose(tcp[:3,:3],wrist[:3,:3]@TOOL_ROTATION,atol=2e-7)

    def test_measured_mount_surfaces_close_without_overlap_and_allow_native_travel(self):
        for scale in (.65, .75, .8):
            installation = measured_installation(scale)
            self.assertAlmostEqual(installation["tilt_deg"], 12., places=5)
            closed = [tool_meshes(scale, side, installation) for side in (0, 1)]
            left = max(mesh.bounds[1,0] for mesh in closed[0].values())
            right = min(mesh.bounds[0,0] for mesh in closed[1].values())
            self.assertAlmostEqual(right-left, installation["minimum_rest_gap_m"], places=7)
            for height in (.035*scale, .055*scale, .073*scale):
                radius, half, gap = contact_geometry(scale, installation, height)
                self.assertGreaterEqual(gap, installation["minimum_rest_gap_m"]-1e-8)
                self.assertLess(gap, .0015)
                center, width, jaw = centered_closure(np.array([[-.0175,0,0],[.0175,0,0]]),
                    np.array([1.,0,0]),radius,half,.003,2*half)
                self.assertGreater(jaw, 0.)
                self.assertLess(jaw, .088)
                self.assertAlmostEqual(jaw+gap, width-.006)

    def test_installation_rejects_stale_scale_source_and_fixture_geometry(self):
        installation = measured_installation(.75)
        validate_installation(installation, .75)
        with self.assertRaises(ValueError):
            validate_installation(installation, .65)
        for key, value in (("source_sha256", "old-source"), ("seat_width_m", .04),
                           ("forward_offset_m", 0.)):
            stale = {**installation, key: value}
            with self.assertRaises(ValueError):
                validate_installation(stale, .75)

    def test_camera_housing_leaves_full_optical_frustum_open(self):
        for scale in (.65, .75, .8):
            case = camera_housing_surface(scale)
            # All sensor edge rays at the production 2.8 mm focal length.
            x, y = np.meshgrid(np.linspace(-2.4/2.8, 2.4/2.8, 17),
                               np.linspace(-2.4/2.8, 2.4/2.8, 17))
            rays = np.column_stack([x.ravel(), y.ravel(), np.ones(x.size)])
            origins = np.tile([0., 0., .006*scale], (len(rays), 1))
            points, _, _ = case.ray.intersects_location(origins, rays, multiple_hits=False)
            self.assertEqual(len(points), 0, "The camera case obstructs its own sensor")

    def test_uniform_scaling_preserves_topology_and_material(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg=ExperimentConfig()
            before=TetMesh.load(cfg.mesh)
            modulus=cfg.material.young_modulus_pa
            camera=np.array(cfg.camera.position_in_finger_m)
            scale_config(cfg,.65,Path(directory))
            after=TetMesh.load(cfg.mesh)
            np.testing.assert_array_equal(after.tets,before.tets)
            np.testing.assert_allclose(after.vertices,before.vertices*.65)
            self.assertAlmostEqual(after.volume,before.volume*.65**3)
            self.assertEqual(cfg.material.young_modulus_pa,modulus)
            np.testing.assert_allclose(cfg.camera.position_in_finger_m,camera*.65)

if __name__=="__main__":
    unittest.main(verbosity=2)
