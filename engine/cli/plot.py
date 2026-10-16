"""View a saved run or export one frame by index/model time; never rerun the PDE."""
import argparse
from pathlib import Path
from engine.visualization.spectral import plot_spectral


def main(argv=None):
    """Parse viewer options, render the archive, and report the actual saved time.

    argv is optional for embedding/tests. User mistakes become concise argparse
    errors. Numerical model time has the units chosen by the model, not playback
    seconds; --fps only affects animations. --time does not interpolate frames.
    """
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True,help='manifest.json or its run directory')
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument('--frame',type=int,help='Zero-based saved-frame index')
    selection.add_argument('--time',type=float,help='Model time; choose a saved frame (default: nearest)')
    parser.add_argument('--time-mode',choices=('nearest','before','after'),default=None,
                        help='Saved-time selection; nearest ties choose earlier time; requires --time')
    parser.add_argument('--event-side',choices=('before','after'),default=None,
                        help='First/last frame at a repeated event time (default: after); requires --time')
    parser.add_argument('--save',type=Path,help='Image for --frame/--time, GIF/MP4 for animation; omit to open viewer')
    parser.add_argument('--fps',type=int,default=10,help='Animation playback frames/second, not model time')
    parser.add_argument('--colorscale',choices=('frame','global'),default='frame',help='Global scale scans all saved frames')
    parser.add_argument('--dimension',type=int,choices=(1,2,3),help='Default: archive dimension')
    parser.add_argument('--grid',type=int,default=40,help='Max points per rendered 3D plane axis')
    parser.add_argument('--slices',type=float,nargs=3,default=(.5,.5,.5),metavar=('X','Y','Z'),help='Slice fractions in [0,1]')
    args=parser.parse_args(argv)
    if args.time is None and (args.time_mode is not None or args.event_side is not None):
        parser.error('--time-mode and --event-side require --time')
    if args.manifest.is_dir():args.manifest=args.manifest/'manifest.json'
    try:
        result=plot_spectral(args.manifest,frame_index=args.frame,time=args.time,
            time_mode=args.time_mode or 'nearest',event_side=args.event_side or 'after',
            save=args.save,fps=args.fps,color_scale=args.colorscale,view_dimension=args.dimension,
            grid_points=args.grid,slice_fractions=args.slices)
    except (ValueError,KeyError,OSError) as exc:
        parser.error(str(exc))
    if result is not None:
        requested='' if result['requested_time'] is None else f"requested_time={result['requested_time']}; "
        print(f"{requested}actual_time={result['actual_time']}; "
              f"frame={result['frame_index']}; segment={result['segment']}"+
              (f'; saved={args.save}' if args.save else ''))


if __name__=='__main__':
    main()
