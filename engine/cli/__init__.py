"""Single command-line entry point; numerical modules are loaded on demand."""
import argparse
import runpy
import sys
from importlib.resources import files
from pathlib import Path

from engine import __version__

COMMANDS = {
    'run': ('engine.cli.run', 'Run a spectral simulation from a JSON config'),
    'plot': ('engine.cli.plot', 'View saved segments or export a frame'),
    'video': ('engine.cli.video', 'Export a saved run to GIF or MP4'),
    'analyze': ('engine.cli.analyze', 'Analyze a segment or compute reference errors'),
    'genealogy': ('engine.cli.genealogy', 'Print source/inheritance trees; export CSV/DOT'),
    'diffusion': ('engine.cli.diffusion', 'Run an exact-solution diffusion benchmark'),
    'finite-volume': ('engine.cli.finite_volume', 'Run the independent finite-volume backend'),
    'plot-archive': ('engine.visualization.archive', 'View a standalone NPZ archive'),
    'integrate': ('engine.numerics.ode', 'Integrate an existing ODE/provider descriptor'),
    'newton': ('engine.numerics.newton', 'Run the configured multistart Newton analysis'),
    'spectrum': ('engine.numerics.spectrum', 'Compute the configured local spectrum'),
    'validate-births': ('engine.cli.validate_births', 'Check stochastic birth-clock statistics'),
}


# List bundled JSON examples or copy one to a new editable file. Exclusive creation
# protects an existing experiment from overwrite.
def export_example(argv):
    parser = argparse.ArgumentParser(prog='spatial-engine examples',
                                     description='List bundled configs, or copy one to an editable JSON file.')
    parser.add_argument('name', nargs='?', help='Example name (with or without .json)')
    parser.add_argument('--output', type=Path, help='New destination file; existing files are protected')
    args = parser.parse_args(argv)
    directory = files('engine').joinpath('config')
    names = sorted(p.name for p in directory.iterdir() if p.name.endswith('.json'))
    if args.name is None:
        if args.output is not None:
            parser.error('--output requires an example name')
        print('\n'.join(names))
        return
    name = args.name if args.name.endswith('.json') else args.name + '.json'
    if name not in names:
        parser.error(f'Unknown example {args.name!r}; run spatial-engine examples to list names')
    if args.output is None:
        parser.error('--output is required when copying an example')
    try:
        # Exclusive creation avoids accidentally overwriting an experiment.
        with args.output.open('xb') as destination:
            destination.write(directory.joinpath(name).read_bytes())
    except OSError as exc:
        parser.error(str(exc))
    print(f'Saved {args.output}')


# Dispatch one installed subcommand lazily, forwarding its arguments and restoring
# sys.argv afterward. This entrypoint is shared by python -m engine.
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    choices = '\n'.join(f'  {name:16} {description}' for name, (_, description) in COMMANDS.items())
    parser = argparse.ArgumentParser(
        prog='spatial-engine', description='Spatial Engine: simulation and archive analysis.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=choices + '\n  examples         List or copy bundled JSON configurations\n\nUse <command> --help for command options.')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('command', nargs='?', choices=[*COMMANDS, 'examples'])
    if not argv or argv[0] in ('-h', '--help', '--version'):
        parser.parse_args(argv)
        if not argv:
            parser.print_help()
        return
    args = parser.parse_args(argv[:1])
    if args.command == 'examples':
        return export_example(argv[1:])
    original = sys.argv
    try:
        sys.argv = [f'spatial-engine {args.command}', *argv[1:]]
        runpy.run_module(COMMANDS[args.command][0], run_name='__main__')
    finally:
        sys.argv = original
