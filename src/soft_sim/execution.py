"""Motion-planning commands and task phases, independent of Isaac/Kit."""
import numpy as np


def layout_target_parameters(layout, label):
    """Read either saved or manager-enriched records without mutating them."""
    return next(({"category": category, **record}
        for group in layout.values() if isinstance(group, dict)
        for category, records in group.items() if isinstance(records, list)
        for record in records if isinstance(record, dict) and record.get("label") == label), {})


def pickup_phases(plan, paths):
    opened, closed = plan["candidate"]["opened"], plan["candidate"]["closed"]
    initial = np.repeat(np.asarray(plan["initial_q"][:-2])[None], 2, axis=0)
    grasp = paths["approach"][-1:][[0, 0]]
    phases = [("settle", .3, initial, opened),
              ("pregrasp", max(.5, (len(paths["pregrasp"])-1)*.02), paths["pregrasp"], opened),
              ("approach", max(1., (len(paths["approach"])-1)*.02), paths["approach"], opened),
              ("close", 2., grasp, closed), ("hold", 1., grasp, closed),
              ("probe_lift", 1.5, paths["probe"], closed),
              ("lift", max(2., (len(paths["lift"])-1)*.02), paths["lift"], closed)]
    if "placement" not in plan:
        return phases + [("verify_hold", 1., paths["lift"][-1:][[0, 0]], closed)]
    phases.extend((name, max(minimum, (len(paths[key])-1)*.02), paths[key], width)
                  for name, key, minimum, width in (
                      ("transfer", "transfer", 2., closed),
                      ("place", "place", 2., closed),
                      ("release", "release", 1.5, opened),
                      ("retreat", "retreat", 1., opened),
                      ("return_home", "return_home", 2., opened)))
    phases.append(("verify_hold", 1., paths["return_home"][-1:][[0, 0]], opened))
    return phases


def trajectory_command(trajectory, duration, elapsed, previous_width, width):
    """Sample a TOPP path and its velocity, with smooth jaw motion in SI units."""
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim != 2 or not len(trajectory) or not np.isfinite(trajectory).all():
        raise ValueError("Expected a finite, nonempty joint trajectory")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Phase duration must be positive")
    u = np.clip(elapsed / duration, 0., 1.)
    if len(trajectory) == 1:
        arm, velocity = trajectory[0], np.zeros(trajectory.shape[1])
    else:
        location = u * (len(trajectory)-1)
        lower = min(int(location), len(trajectory)-2)
        fraction = location-lower
        arm = (1-fraction)*trajectory[lower]+fraction*trajectory[lower+1]
        velocity = (trajectory[lower+1]-trajectory[lower])*(len(trajectory)-1)/duration
    jaw = previous_width + u*u*(3-2*u)*(width-previous_width)
    jaw_velocity = 6*u*(1-u)*(width-previous_width)/duration
    return np.r_[arm, jaw/2, jaw/2], np.r_[velocity, jaw_velocity/2, jaw_velocity/2], float(jaw)


def settled_phase_pairs(rows, phase, settle_s=.25):
    """Only compare adjacent captures inside each occurrence of a hold phase."""
    pairs, start = [], None
    for index, row in enumerate(rows):
        if row["phase"] != phase:
            start = None
            continue
        if start is None:
            start = row["time_s"]
        if (index and rows[index-1]["phase"] == phase
                and rows[index-1]["time_s"] >= start+settle_s):
            pairs.append((index-1, index))
    return pairs


def support_placement_translation(target_vertices, support_vertices, gap_m=.002):
    """Center the unchanged payload over a support and leave a small release gap."""
    if not np.isfinite(gap_m) or gap_m < 0:
        raise ValueError("Release gap must be finite and nonnegative")
    target, support = np.asarray(target_vertices), np.asarray(support_vertices)
    if any(x.ndim != 2 or x.shape[1] != 3 or not len(x) or not np.isfinite(x).all()
           for x in (target, support)):
        raise ValueError("Placement needs finite target and support vertices")
    delta = (support.min(0)+support.max(0)-target.min(0)-target.max(0))/2
    delta[2] = support[:, 2].max()-target[:, 2].min()+gap_m
    return delta
