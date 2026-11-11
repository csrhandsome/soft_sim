"""Prepare pinned source snapshots for TOML editable dependencies, without submodules."""
from pathlib import Path
import argparse
import io
import json
import shutil
import subprocess
import tarfile
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def prepare(name, spec):
    destination = ROOT / spec["path"]
    marker = destination / ".soft-sim-source.json"
    expected = {"url": spec["url"], "rev": spec["rev"]}
    if destination.exists():
        if not marker.exists() or json.loads(marker.read_text()) != expected:
            raise RuntimeError(f"Refusing to overwrite unrecognized source: {destination}")
        print(f"{name}: {spec['rev']} already prepared")
        return
    legacy = ROOT / "RoboDojo/third_party" / ("IsaacLab" if name == "isaaclab" else "curobo")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        staged = Path(temp) / "source"
        staged.mkdir()
        if (legacy / ".git").exists():
            actual = subprocess.check_output(["git", "-C", str(legacy), "rev-parse", "HEAD"], text=True).strip()
            if actual != spec["rev"]:
                raise RuntimeError(f"{legacy}: expected {spec['rev']}, found {actual}")
            # Preserve local patches separately; snapshots always start from the pinned commit.
            blob = subprocess.check_output(["git", "-C", str(legacy), "archive", "HEAD"])
            strip = False
        else:
            url = f"{spec['url']}/archive/{spec['rev']}.tar.gz"
            blob = subprocess.check_output(["curl", "-4", "--fail", "--location", "--silent",
                "--show-error", "--connect-timeout", "10", "--max-time",
                str(spec.get("download-timeout", 1800)), url])
            strip = True
        with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
            for member in archive.getmembers():
                parts = Path(member.name).parts[1:] if strip else Path(member.name).parts
                if not parts:
                    continue
                member.name = str(Path(*parts))
                if member.issym() or member.islnk():
                    # These pinned Python sources require no archive symlinks.
                    continue
                target = (staged / member.name).resolve()
                if not target.is_relative_to(staged.resolve()):
                    raise RuntimeError(f"Unsafe archive member: {member.name}")
                archive.extract(member, staged, filter="data")
        if name == "isaaclab":
            setup = staged / "source/isaaclab/setup.py"
            text = setup.read_text()
            assert '"starlette==0.49.1"' in text
            setup.write_text(text.replace('"starlette==0.49.1"', '"starlette>=0.45.3,<=0.49.1"'))
        if name == "curobo":
            # setuptools-scm needs a version when installing an archive without .git.
            config = staged / "pyproject.toml"
            config.write_text(config.read_text().replace(
                "[tool.setuptools_scm]", '[tool.setuptools_scm]\nfallback_version = "0.0.post1.dev100"'))
        (staged / ".soft-sim-source.json").write_text(json.dumps(expected, indent=2) + "\n")
        shutil.move(str(staged), destination)
    print(f"{name}: prepared {spec['rev']} at {destination}")


def detach():
    for repository in ("RoboDojo", "RoboTwin"):
        repo = ROOT / repository
        if not repo.exists():
            continue
        patch = ROOT / "patches" / (repository.lower() + "-no-submodules.patch")
        applied = subprocess.run(["git", "-C", str(repo), "apply", "--cached", "--reverse", "--check", str(patch)],
                                 capture_output=True)
        if applied.returncode:
            subprocess.run(["git", "-C", str(repo), "apply", "--check", str(patch)], check=True)
            subprocess.run(["git", "-C", str(repo), "apply", "--index", str(patch)], check=True)
    repo = ROOT / "RoboDojo"
    if not repo.exists():
        return
    for name, target in (("IsaacLab", ROOT / ".deps/IsaacLab"), ("curobo", ROOT / ".deps/curobo")):
        path = repo / "third_party" / name
        if path.is_symlink():
            if path.resolve() != target.resolve():
                raise RuntimeError(f"Unexpected dependency symlink: {path}")
        elif not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(Path("../../.deps") / name, target_is_directory=True)
        else:
            raise RuntimeError(f"Back up and move existing source before detaching: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detach", action="store_true", help="Apply checkout patch and add compatibility paths")
    args = parser.parse_args()
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["soft-sim"]
    for name, spec in config["source-repositories"].items():
        prepare(name, spec)
    if args.detach:
        detach()
