"""Export a saved run to GIF/MP4 using the project's Matplotlib renderer.

Run after installing the package. No simulation or RHS is run.
Examples (PowerShell, from the project directory):
    spatial-engine video --manifest runs/sandbox_3d
    spatial-engine video --manifest runs/sandbox_3d --format mp4 --fps 20
    spatial-engine video --manifest runs/sandbox_3d --output movie.gif

By default save animation.gif inside the run directory. MP4 needs FFmpeg
on PATH, or --ffmpeg pointing to ffmpeg.exe. Existing output is protected;
use --overwrite to replace it. FPS controls playback of saved frames, not
physical simulation time. Event boundary frames are kept in archive order.
"""
import argparse
from pathlib import Path
import tempfile
import subprocess


# Validate output/format options and export saved frames to GIF or MP4. FFmpeg is needed
# for MP4; no model integration runs here.
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, required=True,
                        help='manifest.json or its run directory')
    parser.add_argument('--output', type=Path,
                        help='Output .gif/.mp4; default: <run>/animation.<format>')
    parser.add_argument('--format', choices=('gif', 'mp4'),
                        help='Default: infer from --output, otherwise gif')
    parser.add_argument('--fps', type=int, default=12, help='Saved frames per second (default: 12)')
    parser.add_argument('--colorscale', choices=('frame', 'global'), default='frame',
                        help='frame: one render pass; global: first scan all frames for fixed colour limits')
    parser.add_argument('--ffmpeg', type=Path, help='Optional path to ffmpeg.exe for MP4')
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing animation')
    parser.add_argument('--dimension', type=int, choices=(1,2,3), help='Default: archive dimension')
    parser.add_argument('--grid', type=int, default=40, help='Max points per rendered 3D plane axis')
    parser.add_argument('--slices', type=float, nargs=3, default=(.5,.5,.5), metavar=('X','Y','Z'), help='Slice fractions in [0,1]')
    args = parser.parse_args()

    manifest = args.manifest.expanduser().resolve()
    if manifest.is_dir():
        manifest = manifest/'manifest.json'
    if not manifest.is_file():
        parser.error(f'Manifest not found: {manifest}')
    if args.fps <= 0:
        parser.error('--fps must be positive')
    suffix_format = args.output.suffix.lower().lstrip('.') if args.output else None
    if args.output and suffix_format not in ('gif', 'mp4'):
        parser.error('--output must end in .gif or .mp4')
    if args.format and suffix_format and args.format != suffix_format:
        parser.error('--format must match the --output extension')
    video_format = args.format or suffix_format or 'gif'
    output = (args.output.expanduser().resolve() if args.output else
              manifest.parent/f'animation.{video_format}')
    if output.exists() and not args.overwrite:
        parser.error(f'Output already exists: {output}. Choose --output or use --overwrite.')
    if output.is_dir():
        parser.error(f'Output is a directory: {output}')

    # Select a noninteractive backend BEFORE importing the shared renderer.
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.animation import writers
    from engine.visualization.spectral import plot_spectral

    if args.ffmpeg is not None:
        encoder = args.ffmpeg.expanduser().resolve()
        if not encoder.is_file():
            parser.error(f'FFmpeg executable not found: {encoder}')
        matplotlib.rcParams['animation.ffmpeg_path'] = str(encoder)
    if video_format == 'mp4' and not writers.is_available('ffmpeg'):
        parser.error('MP4 needs FFmpeg on PATH or --ffmpeg "C:/path/to/ffmpeg.exe". '
                     'You can export GIF now with --format gif (no FFmpeg needed).')

    temporary = None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        # Encode to a temporary file so a failed/interrupted export cannot
        # destroy a previous completed animation, even with --overwrite.
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.animation-',
                                         suffix='.'+video_format, delete=False) as f:
            temporary = Path(f.name)
        print(f'Exporting {video_format.upper()} at {args.fps} fps: {output}', flush=True)
        plot_spectral(manifest, save=temporary, fps=args.fps, color_scale=args.colorscale,view_dimension=args.dimension,grid_points=args.grid,slice_fractions=args.slices)
        temporary.replace(output)
        print(f'Saved: {output}', flush=True)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.error(f'Export failed: {exc}')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
