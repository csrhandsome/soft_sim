"""Is 80 deformable solver iterations the right number?

A complete episode measures ~34 ms per physics step but cannot say how much of
that is the deformable solve. This group runs the isolated two-finger probe (no
robot, table, objects, contacts or rendering) at several PhysX deformable solver
iteration counts and separates two costs the episode number hides:

* ``fixed_ms`` — per-step work that does not depend on the iteration count
  (scene update, deformable bookkeeping, synchronization, kernel launches),
* ``marginal_ms`` — the marginal cost of one more solver iteration.

It also records how far the deformation drifts from the highest iteration count,
because fewer iterations is only free if the fingers still deform the same way.

Each candidate is measured in alternating rounds, like the CPU benchmark group:
a laptop GPU ramps its clocks during a short run, so single-pass timings on this
card differ by more than the effect being measured. Only the steady-state part of
each run (last ten steps) enters the median.

Run via ``python scripts/run.py test --group gpu-solver``. Kit needs
``OMNI_KIT_ACCEPT_EULA=YES`` like every other command in ``docs/``. Knobs:

SOFT_SIM_SOLVER_ITERATIONS   comma separated candidates, default ``80,40,20``
SOFT_SIM_SOLVER_DURATION_S   simulated seconds per run, default ``1.0``
SOFT_SIM_SOLVER_ROUNDS       alternating rounds per candidate, default ``3``
SOFT_SIM_SOLVER_MIN_FREE_MIB free VRAM required to start, default ``5000``
SOFT_SIM_SOLVER_DRIFT_LIMIT  relative drift still treated as equivalent, default ``.05``
"""
import json
import os
from datetime import datetime, timezone
from statistics import median
import subprocess
import unittest

import numpy as np

from soft_sim.runtime import run
from soft_sim.storage import Storage

#: Accepted spellings of the Kit license flag.
EULA_VALUES = ("yes", "y", "1", "true")


def free_vram_mib():
    """Free device memory, or ``None`` when no driver is reachable."""
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                                 "--format=csv,noheader,nounits"],
                                check=True, capture_output=True, text=True, timeout=10).stdout
        return int(output.split()[0])
    except (FileNotFoundError, subprocess.SubprocessError, IndexError, ValueError):
        return None


def spread(values):
    """Round-to-round spread as a fraction of the median; the honesty check."""
    if len(values) < 2 or not median(values):
        return 0.
    return (max(values) - min(values)) / median(values)


def settled_nodes(state, side):
    """Average of the last quarter of samples; a single instant compares vibration phase."""
    stack = state.get(f"sampled_nodes_{side}")
    if stack is None:
        return state[f"final_nodes_{side}"]
    return stack[-max(1, len(stack) // 4):].mean(axis=0)


def settle_wobble(report, fraction=.25):
    """How much the recorded sag still moves at the end, as a fraction of the sag."""
    sags = [max(side["maximum_displacement_mm"] for side in frame["sides"])
            for frame in report["frames"]]
    tail = sags[-max(1, int(len(sags) * fraction)):]
    return (max(tail) - min(tail)) / max(tail) if tail and max(tail) else 0.


class SolverIterationSweep(unittest.TestCase):
    def test_iteration_count_is_split_into_fixed_and_marginal_cost(self):
        if os.environ.get("OMNI_KIT_ACCEPT_EULA", "").strip().lower() not in EULA_VALUES:
            self.skipTest("Set OMNI_KIT_ACCEPT_EULA=YES; the workers stop at the Kit EULA prompt "
                          "instead of running, the same as every other Isaac Sim command")
        minimum_free = int(os.environ.get("SOFT_SIM_SOLVER_MIN_FREE_MIB", 5000))
        free = free_vram_mib()
        if free is not None and free < minimum_free:
            self.skipTest(f"Only {free} MiB VRAM free (need {minimum_free}): another Isaac Sim "
                          f"instance is running (the probe itself peaks near 7.5 GB on an 8 GB "
                          f"card); rerun once it exits")
        candidates = [int(value) for value in
                      os.environ.get("SOFT_SIM_SOLVER_ITERATIONS", "80,40,20").split(",")]
        self.assertGreaterEqual(len(candidates), 2, "A sweep needs at least two iteration counts")
        duration = float(os.environ.get("SOFT_SIM_SOLVER_DURATION_S", "1."))
        rounds = max(1, int(os.environ.get("SOFT_SIM_SOLVER_ROUNDS", 3)))
        drift_limit = float(os.environ.get("SOFT_SIM_SOLVER_DRIFT_LIMIT", ".05"))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = Storage.create().cache_dir("benchmarks") / f"solver_{stamp}"
        destination.mkdir(parents=True, exist_ok=True)
        print(f"SOLVER_SWEEP_REPORT {destination/'sweep.json'}", flush=True)
        report_path = destination / "sweep.json"

        measurements, progress = {}, []
        for round_index in range(rounds):
            order = candidates if round_index % 2 == 0 else list(reversed(candidates))
            for iterations in order:
                free_now = free_vram_mib()
                if free_now is not None and free_now < minimum_free:
                    self.fail(f"Only {free_now} MiB free before iterations={iterations} round={round_index}: "
                              f"another Isaac Sim instance appeared mid-sweep and would make both "
                              f"processes OOM; rerun when the card is idle")
                folder = destination / f"round_{round_index}_iter_{iterations:03d}"
                process = run("robodojo", "soft_sim.sim.diagnostics",
                              "--iterations", iterations, "--duration", duration,
                              "--output", folder, log_path=destination / f"round_{round_index}_iter_{iterations:03d}.log")
                self.assertTrue(process.ok, f"iteration sweep failed at {iterations}: {process.log_path}")
                report = json.loads((folder / "report.json").read_text())
                state = np.load(folder / "state.npz")
                timing = report["timing"]
                self.assertEqual(timing["steps"], round(duration / report["dt"]))
                self.assertIsNotNone(timing["world_step_ms"], f"no step timing at {iterations}")
                entry = dict(round=round_index, iterations=iterations, report=report,
                             step_ms=timing["world_step_ms"]["last_ten_ms"],
                             mean_ms=timing["world_step_ms"]["mean"],
                             final_nodes=[settled_nodes(state, 0), settled_nodes(state, 1)],
                             settle_wobble=settle_wobble(report),
                             state_path=str(folder / "state.npz"), log=str(process.log_path),
                             gpu=timing.get("gpu") or {},
                             power_w=(timing.get("gpu") or {}).get("power_w", {}).get("mean"))
                measurements.setdefault(iterations, []).append(entry)
                progress.append(dict(round=round_index, iterations=iterations, step_ms=entry["step_ms"],
                                     power_w=entry["power_w"], gpu_util=(entry["gpu"].get("utilisation_percent") or {}).get("mean")))
                report_path.write_text(json.dumps(dict(destination=str(destination), progress=progress), indent=2) + "\n")
                print(f"SOLVER_SWEEP_STEP round={round_index} iterations={iterations} "
                      f"steady_ms={entry['step_ms']:.2f} mean_ms={entry['mean_ms']:.2f} "
                      f"power_w={entry['power_w']} util={(entry['gpu'].get('utilisation_percent') or {}).get('mean')}",
                      flush=True)

        reference_iterations = max(candidates)
        reference = measurements[reference_iterations][-1]
        sag_mm = max(side["maximum_displacement_mm"]
                     for side in reference["report"]["frames"][-1]["sides"])
        #: Beyond this the fingers no longer model the same body; such a run is not
        #: a timing data point for the iteration cost.
        divergence_limit = 1.
        table = []
        for iterations in sorted(candidates):
            entries = measurements[iterations]
            steps = [entry["step_ms"] for entry in entries]
            drift = max(float(np.linalg.norm(nodes - expected, axis=1).max() * 1000)
                        for nodes, expected in zip(entries[-1]["final_nodes"], reference["final_nodes"]))
            table.append(dict(
                iterations=iterations, rounds=len(entries),
                steady_state_ms=float(median(steps)), round_spread=float(spread(steps)),
                steady_state_ms_all=steps,
                mean_ms=float(median([entry["mean_ms"] for entry in entries])),
                maximum_drift_mm=drift, relative_drift=drift / sag_mm if sag_mm else 0.,
                settle_wobble=float(median([entry["settle_wobble"] for entry in entries])),
                power_w=float(median([entry["power_w"] for entry in entries if entry["power_w"]])),
                utilisation_percent=float(median([(entry["gpu"].get("utilisation_percent") or {}).get("mean", 0.)
                                                  for entry in entries])),
                memory_used_mib=float(max((entry["gpu"].get("memory_used_mib") or {}).get("maximum", 0.)
                                          for entry in entries)),
                log=entries[-1]["log"]))
        summary = dict(source="isolated two-finger probe, no robot/contacts/rendering",
                       duration_s=duration, rounds=rounds, physics_dt_s=reference["report"]["dt"],
                       candidates=candidates, reference_iterations=reference_iterations,
                       reference_sag_mm=sag_mm, table=table,
                       drift_limit_relative=drift_limit,
                       drift_metric="mean node position over the last quarter of samples",)

        stable = [row for row in table if row["relative_drift"] <= divergence_limit]
        fitted = [row["iterations"] for row in stable] if len(stable) >= 2 else sorted(candidates)
        counts = np.array(fitted, dtype=float)
        times = np.array([next(row["steady_state_ms"] for row in table if row["iterations"] == count)
                          for count in fitted])
        slope, intercept = np.polyfit(counts, times, 1)
        summary["fit"] = dict(
            candidates_fitted=fitted,
            candidates_excluded=sorted(set(candidates) - set(fitted)),
            divergence_limit_relative=divergence_limit,
            marginal_ms_per_iteration=float(slope), fixed_ms_per_step=float(intercept),
            iterations_explained_percent=float(100 * slope * counts.max() / times.max()))
        equivalent = [row for row in sorted(table, key=lambda row: -row["iterations"])
                      if row["relative_drift"] <= drift_limit]
        cheapest = min(equivalent, key=lambda row: row["iterations"]) if equivalent else None
        summary["recommendation"] = dict(
            minimum_equivalent_iterations=cheapest["iterations"] if cheapest else None,
            steady_state_ms_saved=float(
                next(row["steady_state_ms"] for row in table
                     if row["iterations"] == reference_iterations) - cheapest["steady_state_ms"])
            if cheapest else None,
            percent_saved=float(100 * (1 - cheapest["steady_state_ms"] / next(
                row["steady_state_ms"] for row in table
                if row["iterations"] == reference_iterations))) if cheapest else None)
        report_path.write_text(json.dumps(summary, indent=2) + "\n")

        print("\nITER  steady_ms  spread  drift_mm  drift_rel  wobble  power_W  util%  vram_MiB", flush=True)
        for row in table:
            print(f"{row['iterations']:4d} {row['steady_state_ms']:10.2f} {row['round_spread']:7.2f} "
                  f"{row['maximum_drift_mm']:9.4f} {row['relative_drift']:10.4f} "
                  f"{row['settle_wobble']:7.3f} {row['power_w']:8.1f} {row['utilisation_percent']:6.1f} "
                  f"{row['memory_used_mib']:9.0f}", flush=True)
        print(f"SOLVER_SWEEP_FIT fixed_ms_per_step={intercept:.3f} "
              f"marginal_ms_per_iteration={slope:.4f} "
              f"iterations_explained={summary['fit']['iterations_explained_percent']:.1f}% "
              f"fitted={fitted} excluded={summary['fit']['candidates_excluded']}", flush=True)
        print("SOLVER_SWEEP_RECOMMEND " + json.dumps(summary["recommendation"]), flush=True)

        self.assertGreater(slope, 0.,
                           "More iterations must not be faster; the sweep measured the wrong loop")
        self.assertLess(slope * counts.max(), times.max(),
                        "The per-iteration slope cannot exceed the whole step time")
        self.assertGreater(intercept, 0., "A physics step has work that does not depend on iterations")
        self.assertGreater(sag_mm, 0., "The probe produced no deformation to compare against")


if __name__ == "__main__":
    unittest.main()
