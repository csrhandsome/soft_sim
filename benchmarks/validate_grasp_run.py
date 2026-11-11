"""Validate a recorded planned FEM grasp without restarting Isaac Sim."""
from pathlib import Path
import argparse
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("run",nargs="?",default="data/runs/soft_planned_grasp_55")
args=parser.parse_args()
folder=ROOT/args.run
result=json.loads((folder/"result.json").read_text())
trajectory=json.loads((folder/"trajectory.json").read_text())
assert result["task"]=="grasp_and_hold_after_lift"
assert result["success"] is True
initial=result["initial_object_z_m"]
held=[row for row in trajectory if row["phase"]=="verify_hold"]
assert len(held)>=5, "Missing sustained hold observations"
minimum=min(row["object_position"][2]-initial for row in held)
assert minimum > .06
assert abs(minimum-result["min_held_lift_m"])<1e-8
error=float(np.linalg.norm(np.asarray(result["final_position"][:2])-result["target_xy_m"]))
assert abs(error-result["final_xy_error_m"])<1e-8
assert error<.035
camera_report={}
for phase in ("hold","verify_hold"):
    data=np.load(folder/(phase+"_cameras.npz"),allow_pickle=False)
    rgb,depth=data["rgb"],data["depth_m"]
    assert rgb.shape[:3]==depth.shape and rgb.shape[0]==2 and rgb.shape[-1]==3
    assert rgb.dtype==np.uint8
    std=rgb.std(axis=(1,2,3))
    finite=np.isfinite(depth).sum(axis=(1,2))
    assert np.all(std>1.) and np.all(finite>0)
    assert data["intrinsics"].shape==(2,3,3)
    assert data["camera_to_world"].shape==(2,4,4)
    camera_report[phase]=dict(rgb_std=std.tolist(),finite_depth_pixels=finite.tolist())
report=dict(success=True,max_lift_mm=result["max_lift_m"]*1000,
    final_lift_mm=result["final_lift_m"]*1000,min_held_lift_mm=minimum*1000,
    hold_duration_s=result["verified_hold_duration_s"],xy_drift_mm=error*1000,cameras=camera_report)
(folder/"validation.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
