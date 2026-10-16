"""Read saved frames, reconstruct fields and export reference errors only."""
import csv
import json
from pathlib import Path
import numpy as np
from engine.archive import read_archive, decode
from ..simulation.grid import Grid
from ..galerkin.basis import TensorBasis
from .error_metrics import error_metrics
from .reference_solutions import reference_solution

COLUMNS = ('time', 'field', 'L2', 'rel_L2', 'Linf', 'mass_abs', 'mass_rel')


def analyze_reference(segment, reference, params=None, fields=None, epsilon=1e-12, output=None):
    """Compare every saved time on the archive grid; never evaluate model RHS.

    One reference parameter dictionary is used for every selected field.
    Select one field when references differ between fields.
    """
    segment = Path(segment)
    archive = segment if segment.suffix == '.npz' else segment/'trajectory.npz'
    params = {} if params is None else params
    data = read_archive(archive)
    names = list(dict.fromkeys(str(n) for n in data['fields']))
    selected = names if fields is None else list(dict.fromkeys(fields))
    if not selected or set(selected) - set(names):
        raise ValueError('Select at least one known archived field')
    representation = decode(data, 'metadata_json').get('representation', 'cosine_galerkin')
    basis = None
    if representation == 'cell_average':
        descriptor = decode(data, 'grid_json')
        grid = Grid(descriptor['shape'], descriptor['lengths'])
    elif 'basis_json' in data:
        descriptor = decode(data, 'basis_json')
        basis = TensorBasis.from_descriptor(descriptor)
        grid = basis.grid
    else:
        raise ValueError('Reference comparison needs basis_json or cell_average grid metadata; legacy Newton analysis is unchanged')
    rows = []
    for j, time in enumerate(data['time']):
        ref = reference_solution(reference, grid, float(time), params)
        for name in selected:
            coefficients = data['coefficients'][j, data['fields'] == name]
            u = (basis.reconstruct(coefficients.reshape(basis.shape)) if basis is not None
                 else coefficients.reshape(grid.shape))
            rows.append({'time': float(time), 'field': name,
                         **error_metrics(u, ref, grid, epsilon)})
    report = {'archive': str(archive.resolve()), 'reference': reference,
              'reference_parameters': params, 'epsilon': float(epsilon),
              'representation': representation, 'grid_shape': list(grid.shape),
              'lengths': list(grid.lengths), 'cell_volume': grid.volume,
              'basis': descriptor if basis is not None else None, 'rows': rows}
    if output is not None:
        output = Path(output)
        if output.suffix.lower() not in ('.csv', '.json'):
            raise ValueError('Reference output must be .csv or .json')
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() == '.csv':
            with output.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=COLUMNS)
                writer.writeheader(); writer.writerows(rows)
        else:
            output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return report
