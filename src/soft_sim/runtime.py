"""Environment registry and the only place that starts child interpreters.

Each environment is an independent uv project with its own lock file. Nothing
else in the code base may hard-code ``envs/<name>/.venv/bin/python``.
"""

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys

from soft_sim.config import ROOT

#: Interpreters that own genuinely different dependency sets.
ENVIRONMENT_PROJECTS = {
    "core": ROOT,
    "planning": ROOT / "envs/planning",
    "robodojo": ROOT / "envs/robodojo",
    "dataset": ROOT / "envs/dataset",
    "meshing": ROOT / "envs/meshing",
}

#: CPU-only diagnostics must not initialize CUDA.
CPU_ONLY = ("planning", "dataset", "meshing", "core")


class EnvironmentUnavailable(RuntimeError):
    """A required interpreter or module is missing."""


def package_path():
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Environment:
    name: str

    @property
    def project(self):
        return ENVIRONMENT_PROJECTS[self.name]

    @property
    def interpreter(self):
        override = os.environ.get(f"SOFT_SIM_{self.name.upper()}_PYTHON")
        candidate = Path(override) if override else self.project / ".venv/bin/python"
        if not candidate.is_file():
            raise EnvironmentUnavailable(
                f"Missing {self.name} interpreter at {candidate}. "
                f"Run: bash scripts/sync_envs.sh {self.name}")
        return candidate

    def module_command(self, module, *arguments):
        return [str(self.interpreter), "-m", module, *map(str, arguments)]


def environment(name):
    if name not in ENVIRONMENT_PROJECTS:
        raise EnvironmentUnavailable(f"Unknown environment: {name}")
    return Environment(name)


def child_environment(name, extra=None):
    """Expose the source tree to child processes without requiring an install."""
    env = dict(os.environ)
    source = str(package_path())
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (source, env.get("PYTHONPATH"))))
    env["PYTHONNOUSERSITE"] = "1"
    if name in CPU_ONLY:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env.setdefault("MPLCONFIGDIR", "/tmp/soft-sim-mpl")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
    env.update(extra or {})
    return env


@dataclass
class ProcessResult:
    environment: str
    module: str
    command: list
    returncode: int
    log_path: Path
    started: str
    finished: str

    @property
    def ok(self):
        return self.returncode == 0

    def to_dict(self):
        return dict(environment=self.environment, module=self.module,
                    returncode=self.returncode, log=Path(self.log_path).name,
                    command=self.command, started=self.started, finished=self.finished)

    def check(self):
        if not self.ok:
            raise EnvironmentUnavailable(
                f"{self.environment} {self.module} exited {self.returncode}; see {self.log_path}")
        return self


def run(name, module, *arguments, log_path, extra_env=None, cwd=None):
    """Run ``python -m module`` in a registered environment, recording a log."""
    env = environment(name)
    command = env.module_command(module, *arguments)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")
    with log_path.open("a") as stream:
        stream.write(f"# {started} {name}: {' '.join(map(str, command))}\n")
        stream.flush()
        process = subprocess.run(command, cwd=str(cwd or ROOT), stdout=stream,
                                 stderr=subprocess.STDOUT, env=child_environment(name, extra_env))
    return ProcessResult(name, module, command, process.returncode, log_path,
                         started, datetime.now().isoformat(timespec="seconds"))


def doctor():
    """Report the interpreter/package/source state of every environment."""
    report = {"root": str(ROOT), "source": str(package_path()), "environments": {}}
    for name in ENVIRONMENT_PROJECTS:
        entry = {"project": str(ENVIRONMENT_PROJECTS[name])}
        try:
            entry["interpreter"] = str(environment(name).interpreter)
        except EnvironmentUnavailable as error:
            entry["interpreter"] = None
            entry["error"] = str(error)
        report["environments"][name] = entry
    for relative in ("third_party/RoboDojo", ".deps/IsaacLab", ".deps/curobo",
                     "assets/meshes/soft_finger.stl"):
        report.setdefault("sources", {})[relative] = (ROOT / relative).exists()
    report["python"] = sys.version.split()[0]
    return report
