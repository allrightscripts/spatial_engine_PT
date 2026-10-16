"""Shared 1D/2D/3D rendering of saved physical fields, without solving a PDE."""
import numpy as np
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


# Validate requested display dimension, sampling resolution and slice fractions against
# the archive shape; return the effective view dimension.
def validate_view(shape, dimension, grid_points, slices):
    dimension = len(shape) if dimension is None else dimension
    if dimension not in range(1, len(shape) + 1):
        raise ValueError('Invalid view dimension')
    if not isinstance(grid_points, (int, np.integer)) or grid_points < 2:
        raise ValueError('Viewer grid must be an integer >= 2')
    fractions = np.asarray(slices, float)
    if fractions.shape != (3,) or not np.isfinite(fractions).all() or np.any((fractions < 0) | (fractions > 1)):
        raise ValueError('Slice fractions must be three finite numbers in [0,1]')
    return dimension


# Convert fractional slice locations into bounded cell indices for each spatial axis;
# these are display choices only.
def slice_indices(shape, fractions):
    # Fractions refer to the physical domain; select the containing cell.
    return tuple(min(n - 1, int(f * n)) for n, f in zip(shape, fractions))


def draw_field(ax, u, coords, lengths, dimension, limits, grid_points=80,
               slices=(.5, .5, .5), cell_average=False):
    """3D: three coloured orthogonal planes at cell-centre coordinates.

    2D shows XY at the chosen Z; 1D shows X at the chosen Y/Z.
    grid_points caps only the rendered 3D planes, never solver resolution.
    """
    lo, hi = limits
    indices = slice_indices(u.shape, slices)
    if dimension < u.ndim:
        u = u[(slice(None),) * dimension + indices[dimension:]]
    if dimension == 1:
        if cell_average:
            ax.step(coords[0], u, where='mid')
        else:
            ax.plot(coords[0], u)
        ax.set(xlabel='x1', ylabel='value', xlim=(0, lengths[0]),
               ylim=(lo - .03*(hi-lo), hi + .03*(hi-lo)))
        return None
    if dimension == 2:
        artist = ax.imshow(u.T, origin='lower', aspect='auto',
                           extent=(0, lengths[0], 0, lengths[1]),
                           vmin=lo, vmax=hi, cmap='viridis')
        ax.set(xlabel='x1', ylabel='x2')
        return artist
    norm = Normalize(lo, hi)
    cmap = colormaps['viridis']
    sampled = [np.unique(np.linspace(0, n-1, min(n, grid_points), dtype=int)) for n in u.shape]
    for fixed_axis in range(3):
        a, b = [i for i in range(3) if i != fixed_axis]
        A, B = np.meshgrid(coords[a][sampled[a]], coords[b][sampled[b]], indexing='ij')
        xyz = [None] * 3
        xyz[a], xyz[b] = A, B
        xyz[fixed_axis] = np.full_like(A, coords[fixed_axis][indices[fixed_axis]])
        plane = np.take(u, indices[fixed_axis], axis=fixed_axis)
        plane = plane[np.ix_(sampled[a], sampled[b])]
        ax.plot_surface(*xyz, facecolors=cmap(norm(plane)), shade=False,
                        rstride=1, cstride=1, linewidth=0, antialiased=False)
    ax.set(xlabel='x1', ylabel='x2', zlabel='x3', xlim=(0, lengths[0]),
           ylim=(0, lengths[1]), zlim=(0, lengths[2]))
    ax.set_box_aspect(lengths)
    return ScalarMappable(norm=norm, cmap=cmap)
