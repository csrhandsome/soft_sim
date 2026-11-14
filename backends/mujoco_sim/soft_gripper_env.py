"""MuJoCo 4 backend: volumetric flex, moving pinned bases, native contact labels."""

import xml.etree.ElementTree as ET
import numpy as np
import mujoco as mj

from shared.environment import BaseGripperEnv, ForceReading
from shared.geometry import matrix_to_quat_wxyz, transform_points, rotation_rpy_degrees


def numbers(values):
    return ' '.join(f'{x:.10g}' for x in np.asarray(values).ravel())


class MuJoCoGripperEnv(BaseGripperEnv):
    backend_name = 'mujoco'

    def __init__(self, config):
        super().__init__(config)
        if config.simulation.mujoco_dof != 'full':
            # The reduced model pins its bottom interpolation plane, rather than
            # a finite-thickness band of individual vertex DOFs.
            self.pin_mask = np.isclose(self.mesh.vertices[:, 2], self.mesh.vertices[:, 2].min(), atol=1e-8)
        self.model = self.data = self.renderer = self.viewer = None

    @property
    def backend_metadata(self):
        return dict(version=mj.__version__, soft_body='3D tetrahedral flex',
                    integrator='discrete', force_source='mj_contactForce (test-object contacts only)',
                    boundary_control='kinematic pinned base; angles are imposed, not motor torques',
                    damping_mapping='MuJoCo flex elasticity damping',
                    pinning='bottom interpolation plane' if self.config.simulation.mujoco_dof != 'full' else 'mesh.pin_height_m base band',
                    dof=self.config.simulation.mujoco_dof)

    def build_scene(self, episode):
        self._close_scene()
        if episode.object_shape.startswith('usd:'):
            raise ValueError('MuJoCo cannot directly load USD. Use primitive objects for matched backend experiments, or convert the asset to a MuJoCo mesh first.')
        cfg = self.config
        root = ET.Element('mujoco', model='soft_gripper')
        ET.SubElement(root, 'compiler', angle='radian')
        ET.SubElement(root, 'option', timestep=str(cfg.simulation.timestep_s),
                      gravity=numbers(cfg.simulation.gravity_m_s2), integrator='discrete',
                      solver='Newton', jacobian='sparse', iterations=str(cfg.simulation.solver_iterations), tolerance='1e-6')
        visual = ET.SubElement(root, 'visual')
        ET.SubElement(visual, 'global', offwidth=str(cfg.camera.width), offheight=str(cfg.camera.height))
        # MuJoCo clipping distances are multiplied by statistic.extent.
        ET.SubElement(root, 'statistic', extent='1', center='0 0 .16')
        ET.SubElement(visual, 'map', znear=str(cfg.camera.near_m), zfar=str(cfg.camera.far_m))
        world = ET.SubElement(root, 'worldbody')
        ET.SubElement(world, 'light', pos='0 -.2 .4', dir='0 .3 -1', diffuse='.8 .8 .8')
        ET.SubElement(world, 'light', pos='0 .1 .05', dir='0 0 1', diffuse='.7 .7 .7')
        for side, pose in enumerate(self.camera_poses):
            camera_rotation = pose[:3, :3] @ np.diag([1., -1., -1.])
            fovy = np.rad2deg(2 * np.arctan(cfg.camera.height / (2 * self.intrinsics[side, 1, 1])))
            ET.SubElement(world, 'camera', name=f'bottom_{side}', pos=numbers(pose[:3, 3]),
                          quat=numbers(matrix_to_quat_wxyz(camera_rotation)), fovy=str(fovy))
        ET.SubElement(world, 'geom', name='floor', type='plane', size='.5 .5 .01', rgba='.16 .18 .21 1')
        deformable = ET.SubElement(root, 'deformable')
        mass = self.mesh.volume * cfg.material.density_kg_m3
        for side, name in enumerate(('left', 'right')):
            pos, rotation = self.finger_pose(side)
            base = ET.SubElement(world, 'body', name=f'{name}_base', pos=numbers(pos),
                                 quat=numbers(matrix_to_quat_wxyz(rotation)))
            if cfg.simulation.mujoco_dof == 'full':
                # Free vertices live in WORLD coordinates. Moving the base moves only pinned
                # nodes, rather than teleporting the entire deformable hierarchy.
                initial = transform_points(self.mesh.vertices, pos, rotation)
                bodies, local = [], []
                for i, point in enumerate(initial):
                    if self.pin_mask[i]:
                        bodies.append(f'{name}_base')
                        local.append(self.mesh.vertices[i])
                    else:
                        body_name = f'{name}_v{i}'
                        body = ET.SubElement(world, 'body', name=body_name, pos=numbers(point))
                        ET.SubElement(body, 'inertial', pos='0 0 0', mass=str(mass / len(initial)),
                                      diaginertia='1e-10 1e-10 1e-10')
                        for axis in np.eye(3):
                            ET.SubElement(body, 'joint', type='slide', axis=numbers(axis))
                        bodies.append(body_name)
                        local.append(np.zeros(3))
                flex = ET.SubElement(deformable, 'flex', name=name, dim='3', radius='0.0003',
                                     body=' '.join(bodies), vertex=numbers(local), element=numbers(self.mesh.tets),
                                     rgba=numbers((*cfg.material.color_srgb, 1.)))
            else:
                # Optional reduced-coordinate model for previews; final datasets should
                # compare its convergence against full DOFs and a conforming mesh.
                flex = ET.SubElement(base, 'flexcomp', name=name, type='direct', dim='3',
                                     dof=cfg.simulation.mujoco_dof, cellcount=numbers(cfg.simulation.mujoco_cellcount),
                                     point=numbers(self.mesh.vertices), element=numbers(self.mesh.tets),
                                     mass=str(mass), radius='0.0003',
                                     rgba=numbers((*cfg.material.color_srgb, 1.)))
                # Interpolated DOFs pin grid nodes; pin the complete bottom grid layer.
                cells = cfg.simulation.mujoco_cellcount
                order = 2 if cfg.simulation.mujoco_dof == 'quadratic' else 1
                ET.SubElement(flex, 'pin', gridrange=f'0 0 0 {order*cells[0]} {order*cells[1]} 0')
            ET.SubElement(flex, 'contact', condim='3', selfcollide='none',
                          friction=f'{cfg.material.friction} .005 .0001', solref='.003 1')
            ET.SubElement(flex, 'elasticity', young=str(cfg.material.young_modulus_pa),
                          poisson=str(cfg.material.poisson_ratio), damping=str(cfg.material.damping))
        obj = ET.SubElement(world, 'body', name='object', pos=numbers(episode.object_position_m),
                            quat=numbers(matrix_to_quat_wxyz(rotation_rpy_degrees(episode.object_rpy_deg))))
        if not cfg.object.fixed:
            ET.SubElement(obj, 'freejoint', name='object_joint')
        r, h = episode.object_radius_m, episode.object_height_m
        size = [r] if episode.object_shape == 'sphere' else ([r, r, h / 2] if episode.object_shape == 'box' else [r, h / 2])
        ET.SubElement(obj, 'geom', name='object_surface', type=episode.object_shape, size=numbers(size),
                      contype=str(int(cfg.object.collision)), conaffinity=str(int(cfg.object.collision)),
                      mass=str(cfg.object.mass_kg), friction=f'{cfg.object.friction} .005 .0001',
                      rgba='.8 .25 .3 1', condim='3')
        self.xml = ET.tostring(root, encoding='unicode')
        self.model = mj.MjModel.from_xml_string(self.xml)
        self.data = mj.MjData(self.model)
        self._object_geom = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, 'object_surface')
        self._object_body = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'object')
        self._flex_ids = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_FLEX, name) for name in ('left', 'right')]
        self._base_ids = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, f'{name}_base')
                          for name in ('left', 'right')]
        mj.mj_forward(self.model, self.data)
        if not cfg.simulation.headless:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def set_gripper_state(self, state):
        for side in (0, 1):
            position, rotation = self.finger_pose(side)
            # MuJoCo 4 flex elasticity rejects mocap attachments. Updating the
            # zero-DOF support body's prescribed pose gives the pinned vertices
            # the same boundary displacement without a mocap body.
            index = self._base_ids[side]
            self.model.body_pos[index] = position
            self.model.body_quat[index] = matrix_to_quat_wxyz(rotation)
        for side, pose in enumerate(self.camera_poses):
            self.model.cam_pos[side] = pose[:3, 3]
            self.model.cam_quat[side] = matrix_to_quat_wxyz(pose[:3, :3] @ np.diag([1., -1., -1.]))

    def step_physics(self):
        previous_time = self.data.time
        mj.mj_step(self.model, self.data)
        if self.data.time <= previous_time or not np.isfinite(self.data.qpos).all():
            raise RuntimeError('MuJoCo diverged or auto-reset; episode rejected')
        for name in ('mjWARN_INERTIA', 'mjWARN_BADQPOS', 'mjWARN_BADQVEL', 'mjWARN_BADQACC'):
            if self.data.warning[getattr(mj.mjtWarning, name)].number:
                raise RuntimeError(f'MuJoCo reported {name}; episode rejected')
        if self.viewer is not None:
            self.viewer.sync()

    def get_vertices_world(self):
        # Refresh kinematics AND constraint forces after integration, so all labels
        # and images refer to the same state and timestamp.
        mj.mj_forward(self.model, self.data)
        return np.stack([self.data.flexvert_xpos[int(self.model.flex_vertadr[i]):
                         int(self.model.flex_vertadr[i] + self.model.flex_vertnum[i])].copy() for i in self._flex_ids])

    def get_object_pose(self):
        return np.r_[self.data.xpos[self._object_body], self.data.xquat[self._object_body]].copy()

    def read_force(self):
        normal, vectors = np.zeros(2), np.zeros((2, 3))
        wrench = np.zeros(6)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if contact.efc_address < 0:
                continue
            for side, flex_id in enumerate(self._flex_ids):
                for slot in (0, 1):
                    if contact.flex[slot] == flex_id and contact.geom[1 - slot] == self._object_geom:
                        mj.mj_contactForce(self.model, self.data, index, wrench)
                        normal[side] += max(0., float(wrench[0]))
                        # Contact normal points from slot 0 to slot 1. The returned
                        # wrench acts on slot 1, expressed in the contact frame.
                        sign = -1 if slot == 0 else 1
                        vectors[side] += sign * contact.frame.reshape(3, 3).T @ wrench[:3]
        return ForceReading(normal, vectors, 'native_contact', True, np.full((2, 3), np.nan))

    def render_camera(self):
        if self.renderer is None:
            self.renderer = mj.Renderer(self.model, height=self.config.camera.height, width=self.config.camera.width)
        options = mj.MjvOption()
        options.flags[mj.mjtVisFlag.mjVIS_FLEXSKIN] = True
        rgbs, depths = [], []
        for side in (0, 1):
            self.renderer.update_scene(self.data, camera=f'bottom_{side}', scene_option=options)
            rgbs.append(self.renderer.render().copy())
            self.renderer.enable_depth_rendering()
            try:
                depth = self.renderer.render().copy()
            finally:
                self.renderer.disable_depth_rendering()
            depth[(depth >= self.config.camera.far_m * .999) | (depth <= 0)] = np.nan
            depths.append(depth)
        return np.stack(rgbs), np.stack(depths)

    def _close_scene(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def close(self):
        self._close_scene()
        self.model = self.data = None
