"""Resolve official Isaac asset fragments, local USDs and explicit remote URIs."""

from pathlib import Path

CATALOG = {
    'robotiq_2f85': '/Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85_edit.usd',
    'robotiq_2f85_multiphysics': '/Isaac/Robots_Multiphysics/Robotiq/2F-85/Robotiq_2F_85.usda',
    'panda': '/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd',
    'banana': '/Isaac/Props/YCB/Axis_Aligned/011_banana.usd',
    'marker': '/Isaac/Props/YCB/Axis_Aligned/040_large_marker.usd',
    'soup_can': '/Isaac/Props/YCB/Axis_Aligned/005_tomato_soup_can.usd',
    'cracker_box': '/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd',
}


def resolve_usd_path(value):
    """Call after SimulationApp startup; remote assets may require internet."""
    value = CATALOG.get(value, value)
    if value.startswith('/Isaac/'):
        from isaacsim.storage.native import get_assets_root_path
        root = get_assets_root_path()
        if not root:
            raise RuntimeError('Isaac assets root unavailable. Configure a local assets root or check network access.')
        return root.rstrip('/') + value
    if '://' in value:
        return value
    from shared.config import resolve_asset
    path = resolve_asset(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def add_usd_reference(stage, value, prim_path):
    path = resolve_usd_path(value)
    prim = stage.DefinePrim(prim_path)
    if not prim.GetReferences().AddReference(path):
        raise RuntimeError(f'Failed to add reference: {path}')
    if len(prim.GetPrimStack()) < 2:
        raise RuntimeError(f'USD asset is empty or unresolved: {path}')
    return prim
