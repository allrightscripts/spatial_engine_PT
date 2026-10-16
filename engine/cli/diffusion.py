"""Canonical pure diffusion benchmark through the unchanged segmented runner.

This driver explicitly initializes the saved config's benchmark reference at t=0.
Use the diffusion command (not the run command) to reproduce this benchmark configuration.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from engine.analysis.reference_solutions import reference_solution
from engine.galerkin.run import SegmentedRun


# Build a pure-diffusion run and initialize physical samples from the exact solution
# before projection. Do not reproduce this by running its saved config alone.
def run_benchmark(destination, params, modes=(8,), shape=(128,), lengths=(2.,), t_end=.2):
    if len(modes) != len(shape) or len(lengths) != len(shape) or len(shape) not in (1,2,3):
        raise ValueError('Benchmark requires matching 1D/2D/3D modes, grid and lengths')
    if float(params.get('t0', 0)) != 0:
        raise ValueError('Benchmark driver starts at t0=0')
    D = float(params.get('D', .1))
    c = {'dimension':len(shape), 'seed':2718, 'run_id':'DIFFUSION_REFERENCE',
         'grid':{'shape':list(shape), 'lengths':list(lengths)},
         'dt':.01, 't_end':t_end, 'save_dt':t_end/4,
         'boundary':{'type':'neumann','value':0}, 'model':'cultural_v01',
         'initial_fields':[{'id':'u','family':'benchmark','initial':{'value':0},
                            'parameters':{'diffusion':D,'growth':0,'competition':0,'decay':0}}],
         'events':[], 'split':{'enabled':False}, 'nucleation':{'rate':0},
         'solver':{'backend':'spectral_galerkin','basis':'auto','modes':list(modes),
                   'dealiasing':'none','integrator':'DOP853','rtol':1e-10,'atol':1e-12,'max_step':.01},
         'benchmark_initialization':{'driver':'engine.cli.diffusion',
            'reference':'diffusion_neumann_cosine','parameters':deepcopy(params),
            'note':'driver sets state.u from exact t=0 samples before first Galerkin projection'}}
    run = SegmentedRun(c, destination)
    run.state.u[0] = reference_solution('diffusion_neumann_cosine', run.state.grid, 0, params)
    return run.run()


# Parse reference parameters, axes and resolution, run the canonical diffusion benchmark
# and report its segment count.
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-params',type=Path,required=True)
    parser.add_argument('--modes',type=int,nargs='+',default=[8])
    parser.add_argument('--grid',type=int,nargs='+',default=[128])
    parser.add_argument('--lengths',type=float,nargs='+',default=[2.])
    parser.add_argument('--t-end',type=float,default=.2)
    args=parser.parse_args()
    params=json.loads(args.reference_params.read_text(encoding='utf-8'))
    m=run_benchmark(args.output,params,args.modes,args.grid,args.lengths,args.t_end)
    print(f"Completed {len(m['segments'])} diffusion segment(s): {args.output}")

if __name__=='__main__':main()
