"""Measured link poses -> FEM targets without a per-step host readback.

Imported only inside the Isaac runtime. Double precision matches the existing
SciPy/NumPy transform calculation before the final float32 target conversion.
The PhysX solver, mesh, pin mask and actual articulation poses are unchanged.
"""
from os.path import commonprefix

import numpy as np
import torch
import warp as wp

wp.set_module_options({"enable_backward": False, "fuse_fp": False})


@wp.func
def pose_matrix(poses: wp.array2d(dtype=wp.float32), row: int, w_first: int):
    x = wp.float64(poses[row, 3])
    y = wp.float64(poses[row, 4])
    z = wp.float64(poses[row, 5])
    w = wp.float64(poses[row, 6])
    if w_first == 1:
        w = wp.float64(poses[row, 3])
        x = wp.float64(poses[row, 4])
        y = wp.float64(poses[row, 5])
        z = wp.float64(poses[row, 6])
    length = wp.sqrt(x*x + y*y + z*z + w*w)
    x = x / length
    y = y / length
    z = z / length
    w = w / length
    two = wp.float64(2.0)
    return wp.mat44d(
        x*x-y*y-z*z+w*w, two*(x*y-z*w), two*(x*z+y*w), wp.float64(poses[row, 0]),
        two*(x*y+z*w), -x*x+y*y-z*z+w*w, two*(y*z-x*w), wp.float64(poses[row, 1]),
        two*(x*z-y*w), two*(y*z+x*w), -x*x-y*y+z*z+w*w, wp.float64(poses[row, 2]),
        wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))


@wp.kernel
def measured_mounts(poses: wp.array2d(dtype=wp.float32),
                    links: wp.array(dtype=wp.int32),
                    offsets: wp.array(dtype=wp.mat44d),
                    mounts: wp.array(dtype=wp.mat44d), start: int):
    side = wp.tid()
    body = start + side
    mounts[body] = pose_matrix(poses, links[side], 1) * offsets[body]


@wp.kernel
def node_targets(local: wp.array2d(dtype=wp.vec3d),
                 mounts: wp.array(dtype=wp.mat44d),
                 targets: wp.array2d(dtype=wp.vec4f)):
    body, node = wp.tid()
    p = local[body, node]
    m = mounts[body]
    value = targets[body, node]
    targets[body, node] = wp.vec4f(
        wp.float32(m[0, 0]*p[0] + m[0, 1]*p[1] + m[0, 2]*p[2] + m[0, 3]),
        wp.float32(m[1, 0]*p[0] + m[1, 1]*p[1] + m[1, 2]*p[2] + m[1, 3]),
        wp.float32(m[2, 0]*p[0] + m[2, 1]*p[1] + m[2, 2]*p[2] + m[2, 3]), value[3])


@wp.kernel
def fixture_errors(poses: wp.array2d(dtype=wp.float32),
                   offsets: wp.array(dtype=wp.mat44d),
                   mounts: wp.array(dtype=wp.mat44d),
                   errors: wp.array2d(dtype=wp.float64)):
    body = wp.tid()
    actual = pose_matrix(poses, body, 0) * offsets[body]
    expected = mounts[body]
    d = wp.vec3d(actual[0, 3]-expected[0, 3], actual[1, 3]-expected[1, 3], actual[2, 3]-expected[2, 3])
    errors[body, 0] = wp.length(d)
    maximum = wp.float64(0.0)
    for i in range(3):
        for j in range(3):
            maximum = wp.max(maximum, wp.abs(actual[i, j]-expected[i, j]))
    errors[body, 1] = maximum


class GpuToolFollower:
    """Batch initialized tools; download mounts/fixture errors only on demand.

    Sleep is disabled on these mounted tools. Re-submitting an identical target
    therefore cannot wake a sleeping body. The CPU path retains its exact
    stationary-target suppression for standalone tools that may enable sleep.
    """

    def __init__(self, tools, articulations, finger_links, mount_in_links):
        self.tools, self.articulations = tools, articulations
        self.device = tools[0].targets[0].device
        self.body_count = 2 * len(tools)
        if any(tool.sleep_threshold != 0 for tool in tools):
            raise ValueError("GPU following requires disabled deformable sleep")
        self.links = [torch.tensor([a.body_names.index(name) for name in names],
                     dtype=torch.int32, device=self.device)
                      for a, names in zip(articulations, finger_links)]
        self.offsets = torch.as_tensor(np.concatenate(mount_in_links), dtype=torch.float64, device=self.device)
        self.mounts = torch.empty((self.body_count, 4, 4), dtype=torch.float64, device=self.device)
        local = [points for tool in tools for points in tool.local_nodes]
        if len({points.shape for points in local}) != 1:
            raise ValueError("GPU follower requires matching soft-finger node counts")
        self.local = torch.as_tensor(np.stack(local), dtype=torch.float64, device=self.device)
        sim = tools[0].tensor_sim
        paths = [path for tool in tools for path in tool.paths]
        # The legacy 5.1 binding accepts one wildcard string, even though the
        # newer tensor API documents lists of patterns.
        parents = [p.rsplit("/", 1)[0] for p in paths]
        pattern = commonprefix(parents) + "*/Finger_*"
        self.view = sim.create_soft_body_view(pattern)
        if self.view.count != self.body_count:
            raise RuntimeError("Batched soft-body view did not resolve all fingers")
        # Legacy SoftBodyView has no prim_paths. Resolve and verify its row order
        # once using the exact nodal state, rather than assuming USD traversal order.
        batched = self.view.get_sim_nodal_positions().cpu().numpy()
        individual = [v.get_sim_nodal_positions().cpu().numpy()[0] for t in tools for v in t.views]
        rows = []
        for points in individual:
            matches = [i for i, other in enumerate(batched) if np.array_equal(points, other)]
            if len(matches) != 1 or matches[0] in rows:
                raise RuntimeError("Cannot uniquely identify batched soft-body rows")
            rows.append(matches[0])
        # The target buffer uses PhysX row order; the launch reads logical rows.
        # Reorder constants and mount output through a separate target index map.
        self.rows = torch.tensor(rows, dtype=torch.long, device=self.device)
        self.targets = torch.cat([target for tool in tools for target in tool.targets]).contiguous()
        self.indices = torch.tensor(rows, dtype=torch.int32, device=self.device)
        self.physx_targets = (self.targets if rows == list(range(self.body_count))
                              else torch.empty_like(self.targets))
        self.fixture_view = sim.create_rigid_body_view([p for t in tools for p in t.fixtures.parents])
        expected = [p for t in tools for p in t.fixtures.parents]
        if list(self.fixture_view.prim_paths) != expected:
            raise RuntimeError("Batched fixture rows differ from the requested link order")
        self.errors = torch.empty((self.body_count, 2), dtype=torch.float64, device=self.device)
        self.wp_offsets = wp.from_torch(self.offsets, dtype=wp.mat44d)
        self.wp_mounts = wp.from_torch(self.mounts, dtype=wp.mat44d)
        self.wp_local = wp.from_torch(self.local, dtype=wp.vec3d)
        self.wp_targets = wp.from_torch(self.targets, dtype=wp.vec4f)
        self.wp_errors = wp.from_torch(self.errors)
        self.wp_links = [wp.from_torch(value) for value in self.links]

    def follow(self, *, update_cameras=False):
        # IsaacLab and PhysX run on the current torch CUDA stream. Explicitly
        # launch Warp there as well; do not rely on a default-stream coincidence.
        stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
        for i, articulation in enumerate(self.articulations):
            poses = articulation.data.body_link_pose_w[0]
            wp.launch(measured_mounts, dim=2, inputs=[wp.from_torch(poses), self.wp_links[i],
                      self.wp_offsets, self.wp_mounts, i*2], stream=stream)
        wp.launch(node_targets, dim=self.local.shape[:2],
                  inputs=[self.wp_local, self.wp_mounts, self.wp_targets], stream=stream)
        if self.physx_targets is not self.targets:
            self.physx_targets.index_copy_(0, self.rows, self.targets)
        self.view.set_sim_kinematic_targets(self.physx_targets, self.indices)
        poses = self.fixture_view.get_transforms()
        wp.launch(fixture_errors, dim=self.body_count, inputs=[wp.from_torch(poses),
                  self.wp_offsets, self.wp_mounts, self.wp_errors], stream=stream)
        for i, tool in enumerate(self.tools):
            tool._mounts_gpu = self.mounts[i*2:i*2+2]
            tool._mounts = None
            tool.fixtures._pre_step_error_gpu = self.errors[i*2:i*2+2]
            if update_cameras:
                tool.sync_cameras()
