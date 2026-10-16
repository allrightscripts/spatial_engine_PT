"""Reusable constant field-indexed mobility and opt-in Onsager validation.

Models with nonlinear/spatial mobility retain their own __call__. Birth policy
support is explicit: implement birth_matrix and assign_birth_matrix, or inherit
ConstantFieldMobility when constant coefficients are appropriate to the model.
"""
import numpy as np


# Validate a finite square matrix; optionally require symmetry and positive
# semidefiniteness within tolerance. Never silently project it onto a different matrix.
def validate_onsager(matrix, dissipative=True, tolerance=1e-10):
    L = np.asarray(matrix, float)
    if L.ndim != 2 or L.shape[0] != L.shape[1] or not np.isfinite(L).all():
        raise ValueError('Mobility must be a finite square matrix')
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError('Invalid Onsager tolerance')
    if dissipative:
        if not np.allclose(L, L.T, rtol=0, atol=tolerance):
            raise ValueError('Onsager matrix is not symmetric')
        if np.linalg.eigvalsh((L + L.T) / 2).min() < -tolerance:
            raise ValueError('Onsager matrix is not positive semidefinite')
    return L


class ConstantFieldMobility:
    dissipative_onsager = True

    # Build the constant field-indexed matrix from diagonal diffusion and stable-ID
    # mobility_row entries; return an n by n array.
    def birth_matrix(self, U, fields, grid, t):
        L = np.zeros((len(fields), len(fields)))
        for i, f in enumerate(fields):
            for j, other in enumerate(fields):
                L[i, j] = (f['parameters'].get('diffusion', 0.0) if i == j else
                           f.get('mobility_row', {}).get(other['id'], 0.0))
        return L

    # Add singleton spatial axes to the constant matrix so transport can broadcast it over
    # the physical grid.
    def __call__(self, U, fields, grid, t):
        return self.birth_matrix(U, fields, grid, t).reshape((len(fields), len(fields)) + (1,) * len(grid.shape))

    # Write the expanded matrix back into field diffusion parameters and ID-keyed coupling
    # rows; mutates the supplied metadata.
    def assign_birth_matrix(self, fields, matrix):
        for i, f in enumerate(fields):
            f['parameters']['diffusion'] = float(matrix[i, i])
            f['mobility_row'] = {other['id']: float(matrix[i, j]) for j, other in enumerate(fields) if i != j}
