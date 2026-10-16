"""Main run command: JSON model configuration -> standard segment archives."""
import argparse
import json
import logging
from pathlib import Path
from engine.galerkin.run import SegmentedRun


# User entrypoint: spatial-engine run --config demo.json --output runs/demo
# Config values belong in JSON; model formulas belong in engine/model/.
# Always choose an empty output directory to keep experiments independently auditable.
# Read config JSON, launch a segmented simulation and print completion status. Expected
# input errors use argparse; --debug retains tracebacks.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Empty run directory')
    parser.add_argument('--debug', action='store_true', help='Include exception tracebacks')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    try:
        config = json.loads(args.config.read_text(encoding='utf-8'))
        run = SegmentedRun(config, args.output)
        logging.info('run=%s model=%s dimension=%s grid=%s modes=%s seed=%s integrator=%s',
                     run.manifest['run_id'], config.get('model','cultural_v01'),
                     len(run.state.grid.shape), run.state.grid.shape,
                     config['solver'].get('modes'), config['seed'],
                     config['solver'].get('integrator','Radau'))
        manifest = run.run()
        print(f"{manifest['status']}: {len(manifest['segments'])} segments; "
              f"{len(manifest['events'])} events; {args.output / 'manifest.json'}")
    except (ValueError, KeyError, TypeError, OSError, ImportError, FloatingPointError, NotImplementedError) as exc:
        if args.debug:
            raise
        parser.error(f'{type(exc).__name__}: {exc}')


if __name__ == '__main__':
    main()
