"""Task-specific manipulation sequences, with no simulator or planner imports.

These are execution recipes, not success predicates. Only the upstream task's
RewardManager may declare success. Goals are resolved again from live geometry
before each operation; initial poses are retained only for the swap memory task.
"""
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


BASE_TASKS = (
    "general_pickup", "stack_blocks", "stack_bowls", "arrange_largest_number",
    "sort_nesting_dolls_by_size", "swap_T", "classify_objects",
    "pack_objects_into_box", "build_tower",
    "stack_blocks_by_language", "classify_objects_by_language",
    "put_bottles_into_dustbin", "insert_tubes", "deposit_coin",
    "fill_pen_holder", "cover_blocks", "organize_table",
    "play_stacking_toy", "hang_mugs", "solve_equation",
    "fill_egg_holder", "store_laptop_and_headphones", "store_tools_in_toolbox",
    "imitate_sorting_sequence", "press_by_number", "make_kong",
    "play_tic_tac_toe", "swap_blocks", "match_and_pick_from_conveyor",
    "pick_from_conveyor_by_image", "push_T", "align_blocks",
    "sweep_blocks", "play_Xylophone", "plug_in_charger",
    "insert_key", "fasten_screws", "make_toast",
    "fold_clothes", "pour_liquid_into_cup", "pour_balls_into_vase",
    "pour_by_language",
)

RANDOM_TASKS = (
    "arrange_largest_number_random", "fold_clothes_random", "hang_mugs_random",
    "make_toast_random", "pack_objects_into_box_random", "pour_liquid_into_cup_random",
    "push_T_random", "sort_nesting_dolls_by_size_random", "stack_blocks_random",
    "stack_bowls_random", "store_laptop_and_headphones_random", "sweep_blocks_random",
)

IMPLEMENTED_TASKS = BASE_TASKS + RANDOM_TASKS


def layout_records(layout):
    return {record["label"]: {"category": category, **record}
            for kind in ("Rigid", "Dynamic", "Geometry", "Articulation", "Garment", "Fluid")
            for category, records in layout.get(kind, {}).items()
            for record in records if record.get("label")}


def rotation(quaternion):
    q = np.asarray(quaternion, dtype=float)
    return Rotation.from_quat(np.r_[q[1:], q[0]]).as_matrix()


@dataclass(frozen=True)
class Operation:
    target: str
    mode: str = "pickup"
    references: tuple[str, ...] = ()
    xy: tuple[float, float] | None = None
    orientation_wxyz: tuple[float, ...] | None = None
    slot: int = 0
    count: int = 1
    target_frame: str | None = None
    active_frame: str | None = None

    def to_dict(self):
        return asdict(self)


class TaskAdapter:
    """Bounded operation state and task semantics for the first grab/place family."""

    def __init__(self, task, layout, *, target_label=None):
        if task not in IMPLEMENTED_TASKS:
            raise ValueError(f"No soft-finger action recipe for {task}")
        self.task = task
        self.records = layout_records(layout)
        self.initial_poses = {}
        self.index = 0
        self.operations = self._operations(target_label)
        required = {op.target for op in self.operations}
        required.update(label for op in self.operations for label in op.references)
        missing = required - self.records.keys()
        if missing:
            raise ValueError(f"{task} layout lacks required labels: {sorted(missing)}")

    def _prefix(self, prefix):
        return sorted((label for label in self.records if label.startswith(prefix)),
                      key=lambda label: int(label.removeprefix(prefix).lstrip("_")))

    def _operations(self, target):
        task = self.task.removesuffix("_random")
        if task == "general_pickup":
            return [Operation(target or "target")]
        if target is not None:
            raise ValueError("--target-label applies only to general_pickup; other tasks require all steps")
        if task in ("stack_blocks", "stack_blocks_by_language"):
            return [Operation("block_1", "support", ("block_0",)),
                    Operation("block_2", "support", ("block_1",))]
        if task == "stack_bowls":
            return [Operation("bowl1", "support", ("bowl0",), orientation_wxyz=(1., 0., 0., 0.)),
                    Operation("bowl2", "support", ("bowl1",), orientation_wxyz=(1., 0., 0., 0.))]
        if task == "arrange_largest_number":
            digits = self._prefix("digit")
            if len(digits) not in (4, 5):
                raise ValueError("arrange_largest_number requires four or five digits")
            digits.sort(key=lambda label: self.records[label]["category_idx"] % 10, reverse=True)
            return [Operation(label, "support", (f"mat_{i}",), orientation_wxyz=(1., 0., 0., 0.))
                    for i, label in enumerate(digits)]
        if task == "sort_nesting_dolls_by_size":
            # Match the original reward's descending category_idx % 5 ordering.
            dolls = sorted((f"doll{i}" for i in range(5)),
                           key=lambda label: self.records[label]["category_idx"] % 5, reverse=True)
            return [Operation(label, "table", xy=(-.24 + .12*i, -.12),
                              orientation_wxyz=(1., 0., 0., 0.)) for i, label in enumerate(dolls)]
        if task == "swap_T":
            return [Operation("t0", "table", xy=(0., .04)),
                    Operation("t1", "initial", ("t0",)),
                    Operation("t0", "initial", ("t1",))]
        if task in ("classify_objects", "classify_objects_by_language"):
            operations = []
            for category in range(3):
                labels = self._prefix(f"cat{category}")
                if not labels:
                    raise ValueError(f"{task} has no cat{category} objects")
                operations.extend(Operation(label, "container", (f"basket{category}",),
                                            slot=i, count=len(labels)) for i, label in enumerate(labels))
            return operations
        if task == "pack_objects_into_box":
            # Orient the box first; all four object goals use its live bottom frame.
            return [Operation("box", "table", orientation_wxyz=(2**-.5, 0., 0., 2**-.5)),
                    *[Operation(label, "functional_container", ("box",), slot=i, count=4)
                      for i, label in enumerate(("car", "electric_toothbrush", "hammer", "shoe"))]]
        if task == "build_tower":
            return [Operation("block0", "table", xy=(-.28, .04), orientation_wxyz=(1., 0., 0., 0.)),
                    Operation("block1", "tower_base", ("block0",), slot=0),
                    Operation("block2", "tower_base", ("block0",), slot=1),
                    Operation("block0", "support", ("block1", "block2"), orientation_wxyz=(1., 0., 0., 0.)),
                    Operation("block5", "tower_middle", ("block0",), slot=0),
                    Operation("block6", "tower_middle", ("block0",), slot=1),
                    Operation("block7", "support", ("block5", "block6"), orientation_wxyz=(1., 0., 0., 0.)),
                    Operation("block3", "support", ("block7",), orientation_wxyz=(1., 0., 0., 0.)),
                    Operation("block4", "support", ("block3",), orientation_wxyz=(2**-.5, 2**-.5, 0., 0.))]
        if task == "put_bottles_into_dustbin":
            return [Operation(f"bottle{i}", "container", ("dustbin",), slot=i, count=4)
                    for i in range(4)]
        if task == "insert_tubes":
            return [Operation(f"tube{i}", "container", ("slot",),
                              orientation_wxyz=(2**-.5, 2**-.5, 0., 0.), slot=i, count=3)
                    for i in range(3)]
        if task == "deposit_coin":
            return [Operation("coin0", "functional_slot", ("piggy_bank",),
                              orientation_wxyz=(2**-.5, 2**-.5, 0., 0.),
                              target_frame="passive/center")]
        if task == "fill_pen_holder":
            return [Operation(f"target{i}", "container_tip_down", ("pen_holder",), slot=i, count=4)
                    for i in range(4)]
        if task == "cover_blocks":
            return [
                Operation("cup0", "cover_block", ("red", "green", "blue"), slot=0),
                Operation("cup1", "cover_block", ("red", "green", "blue"), slot=1),
                Operation("cup2", "cover_block", ("red", "green", "blue"), slot=2),
                Operation("cup0", "table", xy=(-.24, -.25)),
                Operation("cup1", "table", xy=(0., -.25)),
                Operation("cup2", "table", xy=(.24, -.25)),
            ]
        if task == "organize_table":
            return [
                Operation("mouse", "support", ("mousemat",), orientation_wxyz=(2**-.5, 0., 0., -2**-.5)),
                Operation("keyboard", "table", xy=(.1, 0.), orientation_wxyz=(1., 0., 0., 0.)),
                Operation("garage", "support", ("cube_cushion",), orientation_wxyz=(1., 0., 0., 0.)),
                Operation("alarm", "support", ("drawer",), orientation_wxyz=(1., 0., 0., 0.)),
            ]
        if task == "play_stacking_toy":
            return [
                Operation("block0", "functional_slot", ("stack_base",), target_frame="passive/pole/0"),
                Operation("block1", "support", ("block0",)),
                Operation("block2", "support", ("block1",)),
                Operation("block3", "support", ("block2",)),
                Operation("block4", "functional_slot", ("stack_base",), target_frame="passive/pole/1"),
                Operation("block5", "support", ("block4",)),
                Operation("block6", "support", ("block5",)),
                Operation("block7", "functional_slot", ("stack_base",), target_frame="passive/pole/2"),
                Operation("block8", "support", ("block7",)),
                Operation("block9", "functional_slot", ("stack_base",), target_frame="passive/pole/3"),
            ]
        if task == "hang_mugs":
            return [Operation(f"mug{i}", "functional_slot", ("cup_holder",),
                              target_frame="passive/handle_support_point",
                              active_frame="active/handle", slot=i, count=3)
                    for i in range(3)]
        if task == "solve_equation":
            return [self._solve_equation_operation()]
        if task == "fill_egg_holder":
            return [*(Operation(f"target{i}", "container", ("egg_holder",), slot=i, count=4)
                      for i in range(4)),
                    Operation("egg_holder", "close_lid")]
        if task == "store_laptop_and_headphones":
            return [
                Operation("headset", "functional_slot", ("stand",),
                          target_frame="passive/support_cradle", active_frame="active/headband_bridge"),
                Operation("laptop", "close_lid"),
                Operation("laptop", "support", ("vertical_laptop_storage_rack",),
                          orientation_wxyz=(2**-.5, 2**-.5, 0., 0.)),
            ]
        if task == "store_tools_in_toolbox":
            return [Operation(tool, "functional_slot", ("toolbox",),
                              target_frame=f"passive/{tool}_slot", active_frame="active/center")
                    for tool in ("pliers", "hammer", "tape_measure", "wrench")]
        if task == "imitate_sorting_sequence":
            return [Operation(f"target{i}", "container", ("basket0",), slot=i, count=5)
                    for i in range(5)]
        if task == "press_by_number":
            n0 = self.records["num0"]["category_idx"] % 10 if "num0" in self.records else 1
            n1 = self.records["num1"]["category_idx"] % 10 if "num1" in self.records else 1
            return [*(Operation("button0", "press") for _ in range(max(1, n0))),
                    Operation("button2", "press"),
                    *(Operation("button1", "press") for _ in range(max(1, n1))),
                    Operation("button2", "press")]
        if task == "make_kong":
            return [Operation("mahjong5_0", "support", ("mahjong0_0",))]
        if task == "play_tic_tac_toe":
            return [Operation(f"player_piece{i}", "functional_slot", ("chessboard",),
                              target_frame="passive/chessboard", slot=i, count=9)
                    for i in range(5)]
        if task == "swap_blocks":
            return [
                Operation("target0", "support", ("mat2",)),
                Operation("button0", "press"),
                Operation("target1", "support", ("mat0",)),
                Operation("target0", "support", ("mat1",)),
            ]
        if task == "match_and_pick_from_conveyor":
            return [Operation("target1", "pickup")]
        if task == "pick_from_conveyor_by_image":
            return [Operation("target", "container", ("basket",))]
        if task == "push_T":
            return [Operation("t", "push", ("target_t",))]
        if task == "align_blocks":
            return [Operation("target", "push", (f"cube{i}",)) for i in range(3)]
        if task == "sweep_blocks":
            return [Operation("broom", "push", ("broom_shovel",))]
        if task == "play_Xylophone":
            return [Operation("mallet", "functional_slot", ("xylophone",),
                              target_frame=f"passive/hit_{i}", active_frame="active/beat")
                    for i in range(4)]
        if task == "plug_in_charger":
            return [Operation("charger", "support", ("socket",),
                              orientation_wxyz=(2**-.5, 2**-.5, 0., 0.))]
        if task == "insert_key":
            return [Operation("key", "support", ("slot",),
                              orientation_wxyz=(1., 0., 0., 0.))]
        if task == "fasten_screws":
            return [Operation(f"nut{i}", "support", (f"bolt{i}",),
                              orientation_wxyz=(1., 0., 0., 0.))
                    for i in range(3)]
        if task == "make_toast":
            return [
                Operation("bread_0", "functional_slot", ("toaster",), target_frame="passive/toast_slot1",
                          orientation_wxyz=(2**-.5, 2**-.5, 0., 0.)),
                Operation("bread_1", "functional_slot", ("toaster",), target_frame="passive/toast_slot2",
                          orientation_wxyz=(2**-.5, 2**-.5, 0., 0.)),
                Operation("toaster", "press", target_frame="passive/toast_botton"),
            ]
        if task == "fold_clothes":
            return [Operation("target", "fold")]
        if task == "pour_balls_into_vase":
            return [Operation("cup", "pour", ("vase",), orientation_wxyz=(2**-.5, 2**-.5, 0., 0.))]
        if task in ("pour_liquid_into_cup", "pour_by_language"):
            src = "bottle" if "bottle" in self.records else ("cup" if "cup" in self.records else next(iter(self.records)))
            dst = "target_cup" if "target_cup" in self.records else ("cup" if "cup" in self.records and src != "cup" else src)
            return [Operation(src, "pour", (dst,) if dst != src else (),
                              orientation_wxyz=(2**-.5, 2**-.5, 0., 0.))]
        raise AssertionError(task)

    def _solve_equation_operation(self):
        ops = ("plus", "minus", "multiplication", "division")
        mat_ids = {int(l[3:]) for l in self._prefix("mat") if l[3:].isdigit()}
        t_ids = {int(l[1:]) for l in self._prefix("t") if l[1:].isdigit()}
        has_mat5 = 5 in mat_ids
        need_ids = {0, 1, 2, 4} | ({5} if has_mat5 else set())
        missing = sorted((mat_ids & need_ids) - t_ids)
        if len(missing) != 1:
            raise ValueError(f"solve_equation expected 1 missing pad, got {missing}")
        missing_id = missing[0]
        vals = {}
        for tid in t_ids & {0, 1, 2, 4, 5}:
            rec = self.records[f"t{tid}"]
            vals[tid] = rec["category"] if rec["category"] in ops else rec["category_idx"] % 10
        def calc(a, op, b):
            if op == "plus": return a + b
            if op == "minus": return a - b if a >= b else None
            if op == "multiplication": return a * b
            if op == "division": return a // b if b != 0 and a % b == 0 else None
            return None
        for cand in (ops if missing_id == 1 else range(10)):
            test = {**vals, missing_id: cand}
            if not all(k in test for k in (0, 1, 2, 4)):
                continue
            pred = calc(test[0], test[1], test[2])
            ans = test[4] * 10 + test[5] if has_mat5 and 5 in test else test[4]
            if pred == ans:
                candidates = ([l for l in self.records if self.records[l]["category"] == cand]
                              if missing_id == 1 else
                              [l for l in self._prefix("num_")
                               if self.records[l]["category"] == "number"
                               and self.records[l]["category_idx"] % 10 == cand])
                if candidates:
                    return Operation(candidates[0], "support", (f"mat{missing_id}",),
                                     orientation_wxyz=(1., 0., 0., 0.))
        raise ValueError("solve_equation could not find valid missing tile in layout")

    @property
    def current(self):
        return self.operations[self.index] if self.index < len(self.operations) else None

    @property
    def recording_labels(self):
        """Stable per-task slots; absent variable-count objects get a validity mask."""
        base = self.task.removesuffix("_random")
        if base == "general_pickup":
            return [self.operations[0].target]
        if base == "arrange_largest_number":
            return [f"digit_{i}" for i in range(5)] + [f"mat_{i}" for i in range(5)]
        if base in ("classify_objects", "classify_objects_by_language"):
            return [f"cat{cat}_{i}" for cat in range(3) for i in range(3)] + [f"basket{i}" for i in range(3)]
        return sorted({op.target for op in self.operations}
                      | {label for op in self.operations for label in op.references})

    def advance(self):
        if self.current is None:
            raise ValueError("Task recipe already exhausted")
        self.index += 1

    def observe_initial(self, scene):
        if not self.initial_poses:
            self.initial_poses = {label: (np.array(obj["position"]), rotation(obj["quaternion_wxyz"]))
                                  for label, obj in scene["task_objects"].items()}
            if self.task == "cover_blocks":
                sorted_blocks = sorted(("red", "green", "blue"),
                                       key=lambda label: self.initial_poses[label][0][0])
                block_to_cup = {block: f"cup{i}" for i, block in enumerate(sorted_blocks)}
                self.operations[3:] = [
                    Operation(block_to_cup["red"], "table", xy=(-.24, -.25)),
                    Operation(block_to_cup["green"], "table", xy=(0., -.25)),
                    Operation(block_to_cup["blue"], "table", xy=(.24, -.25)),
                ]

    def placement(self, scene, scene_dir, *, gap_m=.002):
        """Return world_from_goal_object, or None for a lift-only operation."""
        self.last_placement_report = None
        op = self.current
        if op is None:
            raise ValueError("Task recipe exhausted")
        if op.mode == "pickup":
            return None
        if op.mode in ("press", "close_lid", "fold"):
            obj = scene["task_objects"][op.target]
            goal = np.eye(4)
            goal[:3, :3] = rotation(obj["quaternion_wxyz"])
            goal[:3, 3] = np.asarray(obj["position"], dtype=float)
            return goal
        self.observe_initial(scene)
        objects = scene["task_objects"]
        obj = objects[op.target]
        current_position = np.asarray(obj["position"], dtype=float)
        current_rotation = rotation(obj["quaternion_wxyz"])
        goal_rotation = rotation(op.orientation_wxyz) if op.orientation_wxyz else current_rotation
        if self.task == "insert_tubes":
            from soft_sim.tasks.insert_tubes_geometry import insert_tubes_placement

            goal, report = insert_tubes_placement(
                scene,
                scene_dir,
                op,
                goal_rotation=goal_rotation,
                gap_m=gap_m,
            )
            self.last_placement_report = report
            return goal
        if op.mode == "initial":
            position, goal_rotation = self.initial_poses[op.references[0]]
            goal = np.eye(4)
            goal[:3, :3], goal[:3, 3] = goal_rotation, position
            return goal
        with np.load(Path(scene_dir) / obj["geometry_file"]) as mesh:
            local_vertices = (mesh["vertices"] - current_position) @ current_rotation
        rotated = local_vertices @ goal_rotation.T
        xy = np.asarray(op.xy if op.xy is not None else current_position[:2]).copy()
        bottom = float(scene["table_z"])
        if op.mode == "cover_block":
            sorted_blocks = sorted(op.references, key=lambda label: self.initial_poses[label][0][0])
            block = objects[sorted_blocks[op.slot]]
            if (current_rotation @ np.array([0., 0., 1.]))[2] > 0:
                goal_rotation = Rotation.from_euler("x", np.pi).as_matrix() @ current_rotation
                rotated = local_vertices @ goal_rotation.T
            xy = np.asarray(block["position"][:2], dtype=float)
            bottom = float(scene["table_z"])
        elif op.references:
            points = []
            for label in op.references:
                with np.load(Path(scene_dir) / objects[label]["geometry_file"]) as mesh:
                    points.append(mesh["vertices"].copy())
            support = np.concatenate(points)
            lower, upper = support.min(0), support.max(0)
            xy = (lower[:2] + upper[:2]) / 2
            bottom = float(upper[2])
            if op.mode == "functional_slot":
                ref = objects[op.references[0]]
                frames = ref.get("functional_frames", {}).get(op.target_frame, []) if op.target_frame else []
                if frames:
                    frame = np.asarray(frames[min(op.slot, len(frames) - 1)], dtype=float)
                    xy = frame[:2, 3].copy()
                    bottom = float(frame[2, 3])
                    if op.active_frame:
                        active_frames = obj.get("functional_frames", {}).get(op.active_frame, [])
                        if active_frames:
                            active_rot = np.asarray(active_frames[0], dtype=float)[:3, :3]
                            goal_rotation = frame[:3, :3] @ active_rot.T @ current_rotation
                            rotated = local_vertices @ goal_rotation.T
                else:
                    bottom = float(lower[2]) + .01
            if op.mode in ("tower_base", "tower_middle"):
                # Two columns near either end of the long board, in its local X direction.
                board = objects[op.references[0]]
                axis = rotation(board["quaternion_wxyz"])[:, 0]
                half_span = np.ptp(support @ axis) * .32
                xy += (-1 if op.slot == 0 else 1) * half_span * axis[:2]
                goal_rotation = np.eye(3)
                rotated = local_vertices @ goal_rotation.T
                if op.mode == "tower_base":
                    xy = self.initial_poses[op.references[0]][0][:2].copy()
                    xy += (-1 if op.slot == 0 else 1) * half_span * axis[:2]
                    bottom = float(scene["table_z"])
            if op.mode in ("container", "functional_container", "container_tip_down"):
                container = objects[op.references[0]]
                container_rotation = rotation(container["quaternion_wxyz"])
                container_position = np.asarray(container["position"])
                if op.mode == "container_tip_down":
                    check_frames = obj.get("functional_frames", {}).get("passive/check", [])
                    if check_frames:
                        check_pos = np.asarray(check_frames[0], dtype=float)[:3, 3]
                        axis_world = check_pos - current_position
                    else:
                        extents = np.ptp(local_vertices, axis=0)
                        local_axis = np.eye(3)[int(np.argmax(extents))]
                        axis_world = current_rotation @ local_axis
                    norm = float(np.linalg.norm(axis_world))
                    if norm > 1e-6:
                        align, _ = Rotation.align_vectors([[0., 0., -1.]], [axis_world / norm])
                        goal_rotation = align.as_matrix() @ current_rotation
                        rotated = local_vertices @ goal_rotation.T
                if op.mode == "functional_container":
                    frames = container.get("functional_frames", {}).get("passive/box_bottom", [])
                    if len(frames) != 4:
                        raise ValueError("box lacks its four official box_bottom functional points")
                    floor = np.asarray(frames)[:, :3, 3]
                    # Front-side frames are defined in each asset's active checkpoint.
                    checkpoint = obj.get("functional_frames", {}).get("active/checkpoint", [])
                    if op.target != "shoe" and not checkpoint:
                        raise ValueError(f"{op.target} lacks its official active/checkpoint orientation")
                    frame_rotation = np.asarray(checkpoint[0])[:3, :3] if checkpoint else current_rotation
                    desired_frame = np.eye(3) if op.target == "shoe" else np.diag([-1., -1., 1.])
                    goal_rotation = desired_frame @ frame_rotation.T @ current_rotation
                    rotated = local_vertices @ goal_rotation.T
                else:
                    # Use metadata bbox for the basket floor and a conservative interior margin.
                    floor = support
                local_floor = (floor - container_position) @ container_rotation
                low, high = local_floor.min(0), local_floor.max(0)
                payload = rotated @ container_rotation
                extent = np.ptp(payload, axis=0)
                usable = high[:2] - low[:2] - extent[:2] - .012
                if (usable < 0).any():
                    raise ValueError(f"{op.target} does not fit inside {op.references[0]} with 6 mm wall clearance")
                columns = int(np.ceil(np.sqrt(op.count)))
                rows = int(np.ceil(op.count / columns))
                u = (op.slot % columns + .5) / columns
                v = (op.slot // columns + .5) / rows
                local_xy = (low[:2] + high[:2])/2 + (np.array([u, v])-.5)*usable
                world_slot = container_position + container_rotation @ np.r_[local_xy, low[2]]
                xy = world_slot[:2]
                bottom = float(world_slot[2]) + .006
        goal = np.eye(4)
        goal[:3, :3] = goal_rotation
        goal[:2, 3] = xy - (rotated.min(0)[:2] + rotated.max(0)[:2])/2
        goal[2, 3] = bottom + gap_m - rotated[:, 2].min()
        return goal
