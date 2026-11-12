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
    if destination.is_symlink() or (destination / ".git").exists():
        raise RuntimeError(f"Back up Git metadata before adopting a source snapshot: {destination}")
    if destination.exists():
        if not marker.exists() or json.loads(marker.read_text()) != expected:
            raise RuntimeError(f"Refusing to overwrite unrecognized source: {destination}")
        print(f"{name}: {spec['rev']} already prepared")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        staged = Path(temp) / "source"
        staged.mkdir()
        url = f"{spec['url'].removesuffix('.git')}/archive/{spec['rev']}.tar.gz"
        blob = subprocess.check_output(["curl", "-4", "--fail", "--location", "--silent",
            "--show-error", "--connect-timeout", "10", "--max-time",
            str(spec.get("download-timeout", 1800)), url])
        with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
            for member in archive.getmembers():
                parts = Path(member.name).parts[1:]
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
        patch_snapshot(name, staged)
        (staged / ".soft-sim-source.json").write_text(json.dumps(expected, indent=2) + "\n")
        shutil.move(str(staged), destination)
    print(f"{name}: prepared {spec['rev']} at {destination}")


def patch_snapshot(name, staged):
    if name in ("robodojo", "robotwin"):
        # Git archives contain no submodule contents. Keep snapshots free of
        # registrations and ignore optional legacy source directories.
        (staged / ".gitmodules").unlink(missing_ok=True)
        placeholders = ("third_party/IsaacLab", "third_party/curobo", "XPolicyLab") if name == "robodojo" else ("XPolicyLab",)
        for relative in placeholders:
            path = staged / relative
            if path.is_dir():
                # git archive emits empty directories for gitlinks. Do not
                # remove a directory containing any unexpected source files.
                path.rmdir()
        if name == "robodojo":
            path = staged / ".gitignore"
            text = path.read_text() if path.exists() else ""
            for line in ("/third_party/IsaacLab", "/third_party/curobo", "/XPolicyLab"):
                if line not in text.splitlines():
                    text = text.rstrip("\n") + "\n" + line + "\n"
            path.write_text(text)
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


def detach():
    """Keep the old --detach entry point for compatibility; no Git operations."""
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
    parser.add_argument("--detach", action="store_true", help="Add compatibility source paths (no Git operations)")
    args = parser.parse_args()
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["soft-sim"]
    for name, spec in config["source-repositories"].items():
        prepare(name, spec)
    if args.detach:
        detach()
