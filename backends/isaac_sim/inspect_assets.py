"""Open an official/local gripper or object and report its actual USD frames/joints."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backends.isaac_sim.assets import CATALOG
from backends.isaac_sim.app_launcher import add_livestream_arguments, configure_visualization, create_simulation_app, keep_open
from shared.config import ExperimentConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('asset', nargs='?', default='robotiq_2f85')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--gui', action='store_true')
    add_livestream_arguments(parser)
    parser.add_argument('--export', help='Save a reference stage as .usda')
    parser.add_argument('--report', default='data/generated/isaac_asset_report.json')
    args = parser.parse_args()
    if args.list:
        print(json.dumps(CATALOG, indent=2))
        return
    config = ExperimentConfig()
    config.simulation.headless = not args.gui
    configure_visualization(parser, args, config.simulation)
    config.validate()
    app = create_simulation_app(config.simulation)
    try:
        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics
        from backends.isaac_sim.assets import add_usd_reference, resolve_usd_path
        from shared.config import resolve_asset
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        root = add_usd_reference(stage, args.asset, '/World/Asset')
        for _ in range(10):
            app.update()
        bodies, joints = [], []
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                bodies.append(str(prim.GetPath()))
            if prim.IsA(UsdPhysics.Joint):
                joint = UsdPhysics.Joint(prim)
                joints.append(dict(path=str(prim.GetPath()), type=prim.GetTypeName(),
                                   body0=[str(p) for p in joint.GetBody0Rel().GetTargets()],
                                   body1=[str(p) for p in joint.GetBody1Rel().GetTargets()]))
        report = dict(asset=resolve_usd_path(args.asset), stage_meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage),
                      rigid_bodies=bodies, joints=joints)
        output = resolve_asset(args.report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        if args.export:
            output = resolve_asset(args.export)
            output.parent.mkdir(parents=True, exist_ok=True)
            stage.GetRootLayer().Export(str(output))
        if args.gui or config.simulation.livestream:
            keep_open(app)
    finally:
        app.close()


if __name__ == '__main__':
    main()
