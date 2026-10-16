"""Analyze a segment: Newton/spectrum, or saved-frame reference errors.

This is orchestration, not a replacement Newton or spectrum implementation.
Default: all coefficients, one initial guess from the last stored state.
"""
import argparse
from pathlib import Path
import numpy as np
from engine.archive import read_archive,decode
from engine.numerics.rhs import load_provider
from engine.numerics.newton import newton
from engine.numerics.spectrum import numerical_jacobian


# Newton analysis re-evaluates the stored provider with matching model source.
# Reference-error mode below instead reconstructs saved frames and compares
# samples; it does not rerun the dynamics. Choose the mode deliberately.
def analyze(folder, time=None, start='last', output=None):
    """Analyze one frozen RHS; output defaults to the selected segment for API compatibility."""
    folder=Path(folder);data=read_archive(folder/'trajectory.npz')
    provider=load_provider(str(data['equations_text'].item()))
    autonomous=provider.config.get('autonomous',True)
    if not autonomous and time is None:
        raise ValueError('Nonautonomous model: provide --time for a frozen-time analysis; it is not a flow equilibrium')
    when=float(data['time'][0]) if time is None else float(time)
    guess=data['coefficients'][-1].copy() if start=='last' else np.zeros_like(data['initial_coefficients'])
    F=lambda a:provider.rhs(when,a)
    result=newton(F,guess)
    if result is None:raise RuntimeError('Newton did not converge from this guess; choose another initial point with the original tools')
    point,residual,iterations=result
    J=numerical_jacobian(F,point);eigenvalues,eigenvectors=np.linalg.eig(J)
    output=Path(output) if output is not None else folder/'point_spectrum.npz'
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,point_used=point,residual_inf=np.array(residual),
                        newton_iterations=np.array(iterations),jacobian=J,eigenvalues=eigenvalues,
                        eigenvectors=eigenvectors,analysis_time=np.array(when),
                        frozen_time=np.array(not autonomous),dimensions=np.arange(len(point)))
    return {'dof':len(point),'residual':residual,'iterations':iterations,
            'stable':int(np.sum(eigenvalues.real < -1e-8)),
            'unstable':int(np.sum(eigenvalues.real > 1e-8)),
            'neutral':int(np.sum(np.abs(eigenvalues.real)<=1e-8))}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--segment',type=Path,required=True)
    parser.add_argument('--time',type=float)
    parser.add_argument('--start',choices=('last','zero'),default='last')
    parser.add_argument('--output',type=Path,help='Newton: required .npz; reference: .csv/.json (default: segment/errors.csv)')
    parser.add_argument('--reference',help='Registered exact/reference solution name')
    parser.add_argument('--reference-params',type=Path,help='JSON reference parameter object')
    parser.add_argument('--field',action='append',help='Compare this field (repeatable; default: all)')
    parser.add_argument('--epsilon',type=float,default=1e-12,help='Positive relative-error denominator regularizer')
    args=parser.parse_args()
    if args.reference:
        if args.time is not None or args.start != 'last':
            parser.error('--time/--start are Newton options; reference mode compares all saved frames')
        import json
        from engine.analysis.reference_errors import analyze_reference
        params=json.loads(args.reference_params.read_text(encoding='utf-8')) if args.reference_params else {}
        destination=args.output or args.segment/'errors.csv'
        report=analyze_reference(args.segment,args.reference,params,args.field,args.epsilon,destination)
        print(f"Saved {len(report['rows'])} frame/field error rows: {destination}")
        for field in dict.fromkeys(r['field'] for r in report['rows']):
            final=next(r for r in reversed(report['rows']) if r['field']==field)
            print(final)
    else:
        if args.reference_params or args.field or args.epsilon != 1e-12:
            parser.error('--reference-params/--field/--epsilon require --reference')
        if args.output is None:
            parser.error('--output is required for Newton/spectrum analysis')
        print(analyze(args.segment,args.time,args.start,args.output))
