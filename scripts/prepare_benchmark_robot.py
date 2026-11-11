"""Make a relocatable URDF using the meshes included in RoboDojo's USD asset."""
from pathlib import Path
import os
import shutil
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]
assets = ROOT / "RoboDojo/Assets/Robots/franka"
out = ROOT / "data/generated/robodojo_franka_local"
out.mkdir(parents=True, exist_ok=True)
tree = ET.parse(assets / "panda.urdf")
for link in tree.getroot().findall("link"):
    for purpose, stem in (("visual", "visuals"), ("collision", "collisions")):
        for group in list(link.findall(purpose)):
            for mesh in group.findall("geometry/mesh"):
                choices = sorted((assets / "meshes").glob(link.attrib["name"] + "_" + stem + "*.obj"))
                if not choices:
                    if link.attrib["name"].startswith("panda_"):
                        raise FileNotFoundError(link.attrib["name"] + " " + purpose)
                    # This dataset omits the original camera accessory meshes.
                    # Keep their frames but omit missing geometry in this test URDF.
                    link.remove(group)
                else:
                    mesh.set("filename", os.path.relpath(choices[0], out))
tree.write(out / "panda.urdf", encoding="utf-8", xml_declaration=True)
shutil.copyfile(assets / "panda.srdf", out / "panda.srdf")
print(out / "panda.urdf")
