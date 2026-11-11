"""Explicitly approximate force from volumetric deformation when native contacts are unavailable."""

import numpy as np


class QuasistaticForceEstimator:
    """Corotated linear FEM balance, not a simulator-native contact sensor.

    Assumes negligible acceleration, no other contacts, isotropic linear material.
    Returns total external force on the free part of the finger after gravity
    compensation. It cannot separate friction from normal contact magnitude.
    """

    def __init__(self, vertices, tets, pin_mask, material, gravity):
        self.tets = tets
        x = vertices[tets]
        dm = np.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]], axis=-1)
        self.inverse = np.linalg.inv(dm)
        self.volume = np.linalg.det(dm) / 6
        gradients = np.empty((len(tets), 4, 3))
        gradients[:, 1:] = self.inverse
        gradients[:, 0] = -self.inverse.sum(axis=1)
        self.gradients = gradients
        self.pin_mask = pin_mask
        self.pinned_in_elements = pin_mask[tets]
        nodal_masses = np.zeros(len(vertices))
        for corner in range(4):
            np.add.at(nodal_masses, tets[:, corner], self.volume * material.density_kg_m3 / 4)
        self.free_weight = nodal_masses[~pin_mask].sum() * np.asarray(gravity)
        self.mu = material.young_modulus_pa / (2 * (1 + material.poisson_ratio))
        self.lam = material.young_modulus_pa * material.poisson_ratio / ((1 + material.poisson_ratio) * (1 - 2 * material.poisson_ratio))

    def estimate(self, vertices):
        x = vertices[self.tets]
        ds = np.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]], axis=-1)
        f = ds @ self.inverse
        if np.any(np.linalg.det(f) <= 0):
            return np.full(3, np.nan)
        u, _, vt = np.linalg.svd(f)
        rotation = u @ vt
        trace = np.trace(np.swapaxes(rotation, 1, 2) @ f, axis1=1, axis2=2) - 3
        piola = 2 * self.mu * (f - rotation) + self.lam * trace[:, None, None] * rotation
        nodal_internal = -self.volume[:, None, None] * np.einsum('eij,ekj->eki', piola, self.gradients)
        return (nodal_internal * self.pinned_in_elements[:, :, None]).sum(axis=(0, 1)) - self.free_weight
