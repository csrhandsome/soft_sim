"""Cache selected official YCB USDs and their textures via the configured asset API."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import charset_normalizer
from isaacsim import SimulationApp
app=SimulationApp({'headless':True})
try:
    from isaacsim.storage.native import get_assets_root_path
    from pxr import UsdUtils
    import omni.client
    root=get_assets_root_path()
    base=root+'/Isaac/Props/YCB/Axis_Aligned/'
    out=Path('data/scene_assets/ycb');out.mkdir(parents=True,exist_ok=True)
    report=[]
    for stem in ('003_cracker_box','005_tomato_soup_can','011_banana'):
        uri=base+stem+'.usd'
        layers,assets,unresolved=UsdUtils.ComputeAllDependencies(uri)
        dependencies={uri}
        dependencies.update(layer.realPath for layer in layers if layer.realPath.startswith(base))
        dependencies.update(str(a) for a in assets if str(a).startswith(base))
        # MDL modules such as OmniPBR.mdl come with Isaac Sim, not the asset pack.
        for src in sorted(dependencies):
            dst=out/src[len(base):]
            dst.parent.mkdir(parents=True,exist_ok=True)
            result=omni.client.copy(src,str(dst.resolve()))
            if result not in (omni.client.Result.OK,omni.client.Result.ERROR_ALREADY_EXISTS):
                raise RuntimeError(f'Cannot cache {src}: {result}')
            print('CACHED',dst,dst.stat().st_size,flush=True)
        report.append({'asset':stem,'source':uri,'dependencies':sorted(dependencies),'unresolved':list(map(str,unresolved))})
    (out/'sources.json').write_text(json.dumps(report,indent=2))
    print('ASSETS_READY',len(report),flush=True)
except Exception:
    import traceback;traceback.print_exc()
finally:
    app.close()
