"""Compare the serialized planning URDF with the shared simulation mount math."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import mplib
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"benchmarks")]
from plan_soft_grasp import make_urdf, section_radius
from shared.config import ExperimentConfig
from shared.mesh import load_stl
from shared.tool_mount import mounts_from_tool

class FrameIntegration(unittest.TestCase):
    def test_planner_and_simulation_mounts_match(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"data/generated") as directory:
            out=Path(directory)
            urdf=make_urdf(out,.055,.075,.008)
            names=["panda_joint"+str(i) for i in range(1,8)]+["panda_finger_joint1","panda_finger_joint2"]
            planner=mplib.Planner(urdf=urdf,srdf=out/"soft_franka.srdf",move_group="soft_grasp_tcp",user_joint_names=names)
            cfg=ExperimentConfig().gripper
            cfg.profile="panda"
            cfg.open_jaw_width_m=.08
            cfg.adapter_offset_m=.008
            kin=planner.pinocchio_model
            for width in (.08,.043-.016):
                q=np.array([.2,-.5,.1,-2.2,-.1,1.8,.8,width/2,width/2])
                kin.compute_forward_kinematics(q)
                hand=kin.get_link_pose(planner.link_name_2_idx["panda_hand"]).to_transformation_matrix()
                world_tool=hand.copy()
                world_tool[:3,:3]=hand[:3,:3]@Rotation.from_euler("z",90,degrees=True).as_matrix()
                world_tool[:3,3]=(hand@np.array([0,0,.075,1]))[:3]
                expected=mounts_from_tool(cfg,world_tool,width)
                for i,name in enumerate(("soft_left","soft_right")):
                    actual=kin.get_link_pose(planner.link_name_2_idx[name]).to_transformation_matrix()
                    np.testing.assert_allclose(actual,expected[i],atol=2e-7)
                tcp=kin.get_link_pose(planner.link_name_2_idx["soft_grasp_tcp"]).to_transformation_matrix()
                np.testing.assert_allclose(tcp[:3,3],(hand@np.array([0,0,.13,1]))[:3],atol=2e-7)
                np.testing.assert_allclose(tcp[:3,:3],world_tool[:3,:3],atol=2e-7)

    def test_stl_contact_band_has_correct_units(self):
        vertices,faces=load_stl(ROOT/"data/assetes/soft_finger.stl")
        self.assertTrue(.088 < vertices[:,2].max() < .089)
        radius=section_radius(vertices,faces,.055)
        self.assertTrue(.009 < radius < .011)
        self.assertGreater(radius,section_radius(vertices,faces,.073))

if __name__=="__main__":
    unittest.main(verbosity=2)
