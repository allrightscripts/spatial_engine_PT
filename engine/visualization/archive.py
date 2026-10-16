# Reconstruct 1D/2D/3D spatial fields from the RK4 trajectory archive.
# Requires numpy and matplotlib. Run: python visualize_trajectory.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np


# Paths without a drive/root are relative to this script, not the terminal.
ARCHIVE_PATH = "trajectory.npz"
VIEW_DIMENSION = None  # None = archive dimension; or 1, 2, 3
GRID_POINTS = 80  # per visible axis; 3D is capped at 40 for responsiveness
SLICE_FRACTIONS = (0.5, 0.5, 0.5)  # fixed x2/x3 for 1D; fixed x3 for 2D
FPS = 12
COLOR_SCALE = "frame"  # "frame" reveals detail; "global" keeps one scale over time


# Expand a user path and resolve it relative to the current working directory, not the
# package location.
def resolve_path(value):
    path = Path(value).expanduser()
    return path.expanduser().resolve()


# Read the legacy cosine archive layout and return time/coefficient/field/mode/length
# arrays. Other representations use plot_archive's dispatch.
def read_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        required = {"time", "coefficients", "fields", "modes", "parameters_json"}
        if missing := required - set(archive.files):
            raise ValueError(f"Missing archive arrays: {sorted(missing)}")
        metadata = json.loads(archive['metadata_json'].item()) if 'metadata_json' in archive else {}
        if metadata.get('representation', 'cosine_galerkin') != 'cosine_galerkin':
            raise ValueError('Use plot_archive or trajectory_archive for grid representation')
        times = np.asarray(archive["time"], dtype=float)
        coefficients = np.asarray(archive["coefficients"], dtype=float)
        fields = np.asarray(archive["fields"], dtype=str)
        modes = np.asarray(archive["modes"], dtype=int)
        parameters = json.loads(archive["parameters_json"].item())
    if (times.ndim != 1 or coefficients.ndim != 2 or modes.ndim != 2
            or fields.ndim != 1 or coefficients.shape != (len(times), len(fields))
            or modes.shape[0] != len(fields) or modes.shape[1] not in (1, 2, 3)):
        raise ValueError("Time, coefficient, field, or mode array shapes disagree")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(coefficients)):
        raise ValueError("Archive contains non-finite times or coefficients")
    dimension = modes.shape[1]
    lengths = []
    for i in range(1, dimension + 1):
        length = float(parameters[f"Lx{i}"])
        if not np.isfinite(length) or length <= 0:
            raise ValueError(f"Invalid Lx{i} in archive")
        lengths.append(length)
    return times, coefficients, fields, modes, lengths


# Prepare legacy cosine reconstruction for the requested view. Return coordinates,
# field-to-column masks and a callback taking (coefficient row, field ID).
def make_reconstructor(fields, modes, lengths, view_dimension, grid_points, slice_fractions=SLICE_FRACTIONS):
    dimension = len(lengths)
    if type(view_dimension) is not int or not 1 <= view_dimension <= dimension:
        raise ValueError(f"View dimension must be 1..{dimension} for this archive")
    if grid_points < 2:
        raise ValueError("GRID_POINTS must be at least 2")
    if len(slice_fractions) != 3 or any(not 0 <= frac <= 1 for frac in slice_fractions):
        raise ValueError("SLICE_FRACTIONS must be between 0 and 1")
    n = min(grid_points, 40) if view_dimension == 3 else grid_points
    coords = [np.linspace(0, length, n) for length in lengths[:view_dimension]]
    matrices = [np.cos(np.pi * np.outer(coords[axis] / lengths[axis], modes[:, axis]))
                for axis in range(view_dimension)]
    slice_factor = np.ones(len(fields))
    for axis in range(view_dimension, dimension):
        slice_factor *= np.cos(np.pi * modes[:, axis] * slice_fractions[axis])
    masks = {field: np.where(fields == field)[0] for field in dict.fromkeys(fields)}
    space_labels = "ijk"[:view_dimension]
    subscripts = "m," + ",".join(label + "m" for label in space_labels) + "->" + space_labels

    # Reconstruct the selected field's stored coefficients on the prepared plotting
    # coordinates; never advance the simulation.
    def reconstruct(frame, field):
        ids = masks[field]
        basis = [matrix[:, ids] for matrix in matrices]
        return np.einsum(subscripts, frame[ids] * slice_factor[ids], *basis,
                         optimize=True)

    return coords, masks, reconstruct


# Bound each field by its zero-mode baseline plus/minus the absolute amplitudes
# of nonzero cosine modes over saved frames, then pad the range for plotting.
def field_limits(coefficients, modes, field_indices):
    series = coefficients[:, field_indices]
    is_zero = np.all(modes[field_indices] == 0, axis=1)
    baseline = series[:, is_zero].sum(axis=1)
    amplitude = np.abs(series[:, ~is_zero]).sum(axis=1)
    lo = float(np.min(baseline - amplitude))
    hi = float(np.max(baseline + amplitude))
    pad = max((hi - lo) * 0.03, 1e-9)
    return lo - pad, hi + pad


# Inspect representation and route to spectral, grid or legacy cosine rendering. Cell
# averages must never be interpreted as Fourier coefficients.
def plot_archive(path, view_dimension=None, grid_points=80, frame_index=None,
                 save=None, fps=12, color_scale="frame", slice_fractions=SLICE_FRACTIONS):
    # Representation dispatch: never interpret cell averages as cosine modes.
    with np.load(path, allow_pickle=False) as probe:
        metadata = json.loads(probe['metadata_json'].item()) if 'metadata_json' in probe else {}
        has_basis = 'basis_json' in probe
    if metadata.get('representation') == 'tensor_galerkin' or (has_basis and metadata.get('representation') == 'cosine_galerkin'):
        from engine.visualization.spectral import plot_spectral
        return plot_spectral(path, view_dimension, grid_points, frame_index,
                             save, fps, color_scale, slice_fractions)
    if metadata.get('representation') == 'cell_average':
        from engine.visualization.viewer import plot_grid_archive
        return plot_grid_archive(path, view_dimension, grid_points, frame_index,
                                 save, fps, color_scale, slice_fractions)
    times, coefficients, fields, modes, lengths = read_archive(path)
    dimension = len(lengths)
    view_dimension = dimension if view_dimension is None else view_dimension
    if color_scale not in ("frame", "global"):
        raise ValueError("COLOR_SCALE must be 'frame' or 'global'")
    coords, masks, reconstruct = make_reconstructor(
        fields, modes, lengths, view_dimension, grid_points, slice_fractions)
    names = list(masks)
    figure = plt.figure(figsize=(6 * len(names), 5.4))
    axes = [figure.add_subplot(1, len(names), i + 1,
             projection="3d" if view_dimension == 3 else None)
            for i in range(len(names))]
    color_map = plt.get_cmap("viridis")
    limits = {name: field_limits(coefficients, modes, mask)
              for name, mask in masks.items()}
    mappings = {}
    if view_dimension > 1:
        for axis, name in zip(axes, names):
            mapping = ScalarMappable(norm=Normalize(*limits[name]), cmap=color_map)
            mappings[name] = mapping
            figure.colorbar(mapping, ax=axis, shrink=0.7, pad=0.08)

    midplanes = None
    if view_dimension == 3:
        midx, midy, midz = [min(len(x)-1, int(f*len(x))) for x,f in zip(coords,slice_fractions)]
        x1, x2, x3 = coords
        yy, zz = np.meshgrid(x2, x3, indexing="ij")
        xx, zz2 = np.meshgrid(x1, x3, indexing="ij")
        xx2, yy2 = np.meshgrid(x1, x2, indexing="ij")
        midplanes = ((np.full_like(yy, x1[midx]), yy, zz),
                     (xx, np.full_like(xx, x2[midy]), zz2),
                     (xx2, yy2, np.full_like(xx2, x3[midz])))

    # Render one saved legacy frame, updating titles/color limits and artists for the
    # animation or image exporter.
    def draw(index):
        for axis, name in zip(axes, names):
            axis.clear()
            values = reconstruct(coefficients[index], name)
            if color_scale == "frame":
                span = float(np.max(values) - np.min(values))
                pad = max(span * 0.03, abs(float(np.mean(values))) * 0.01, 1e-9)
                lo, hi = float(np.min(values) - pad), float(np.max(values) + pad)
            else:
                lo, hi = limits[name]
            if view_dimension > 1:
                mappings[name].set_clim(lo, hi)
            if view_dimension == 1:
                axis.plot(coords[0], values, color="#1261a0", linewidth=2)
                axis.set_xlim(0, lengths[0])
                axis.set_ylim(lo, hi)
                axis.set_xlabel("x1")
                axis.set_ylabel(name)
            elif view_dimension == 2:
                axis.imshow(values.T, origin="lower", aspect="auto", cmap=color_map,
                            vmin=lo, vmax=hi,
                            extent=(0, lengths[0], 0, lengths[1]))
                axis.set_xlabel("x1")
                axis.set_ylabel("x2")
            else:
                # Three perpendicular midplanes display the interior of a 3D field.
                normalizer = Normalize(lo, hi)
                planes = (values[midx, :, :], values[:, midy, :], values[:, :, midz])
                for (px, py, pz), plane in zip(midplanes, planes):
                    axis.plot_surface(px, py, pz, facecolors=color_map(normalizer(plane)),
                                      shade=False, linewidth=0, antialiased=False,
                                      rstride=1, cstride=1)
                axis.set(xlim=(0, lengths[0]), ylim=(0, lengths[1]),
                         zlim=(0, lengths[2]), xlabel="x1", ylabel="x2", zlabel="x3")
                axis.set_box_aspect(lengths)
            axis.set_title(f"{name}   t = {times[index]:.4g}")
        return axes

    if frame_index is not None:
        if not 0 <= frame_index < len(times):
            raise ValueError(f"Frame must be in [0, {len(times) - 1}]")
        draw(frame_index)
        if save:
            figure.savefig(save, dpi=150)
            print(f"Saved image: {save}")
        else:
            plt.show()
    else:
        animation = FuncAnimation(figure, draw, frames=len(times), interval=1000 / fps,
                                  repeat=True)
        if save:
            suffix = Path(save).suffix.lower()
            if suffix == ".gif":
                animation.save(save, writer="pillow", fps=fps)
            elif suffix == ".mp4":
                animation.save(save, writer="ffmpeg", fps=fps)
            else:
                raise ValueError("Animation output must end in .gif or .mp4")
            print(f"Saved animation: {save}")
        else:
            plt.show()
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View Galerkin NPZ archive in 1D, 2D or 3D")
    parser.add_argument("--archive", required=True, help="Input NPZ path")
    parser.add_argument("--dimension", type=int, default=VIEW_DIMENSION, choices=(1, 2, 3))
    parser.add_argument("--frame", type=int, help="Plot one saved frame; omit for animation")
    parser.add_argument("--save", help="Save a frame as PNG, or animation as GIF/MP4")
    parser.add_argument("--grid", type=int, default=GRID_POINTS)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--colorscale", default=COLOR_SCALE,
                        choices=("frame", "global"))
    parser.add_argument('--slices', type=float, nargs=3, default=SLICE_FRACTIONS, metavar=('X','Y','Z'))
    args = parser.parse_args()
    source = Path(args.archive).expanduser().resolve()
    if not source.is_file():
        parser.error(f"Archive not found: {source}")
    if args.fps <= 0:
        parser.error("FPS must be positive")
    destination = Path(args.save).expanduser().resolve() if args.save else None
    plot_archive(source, args.dimension, args.grid, args.frame, destination,
                 args.fps, args.colorscale, args.slices)
