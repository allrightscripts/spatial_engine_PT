"""Stable numeric trajectory contract, shared by Galerkin and grid solvers.

A column is a DOF, NOT a physical field. `fields[column]` and `modes[column]`
identify it. Legacy archives without representation metadata mean cosine modes.
The module depends only on NumPy and the Python standard library.
"""
# PUBLIC IO BOUNDARY: use read_archive/write_archive instead of ad-hoc pickle
# or object arrays. Add optional metadata without changing required array keys.
# Existing archives and independent viewers rely on this contract.
import json
import os
from pathlib import Path
import tempfile
import numpy as np

SCHEMA_VERSION = '1.0'
CORE = {'time', 'coefficients', 'fields', 'modes', 'parameters_json', 'metadata_json'}


def json_array(value):
    """Portable scalar Unicode JSON; reject NaN/Infinity and Python objects."""
    return np.array(json.dumps(value, ensure_ascii=False, allow_nan=False))


# Decode a scalar JSON array from an archive dictionary; use default when the key is
# absent. Does not evaluate Python code.
def decode(data, key, default=None):
    return json.loads(str(data[key].item())) if key in data else default


def validate(data):
    """Validate legacy and versioned payloads without executing model code."""
    missing = CORE - data.keys()
    if missing:
        raise ValueError(f'Missing archive keys: {sorted(missing)}')
    for key, value in data.items():
        a = np.asarray(value)
        if a.dtype.hasobject:
            raise ValueError(f'Object array forbidden: {key}')
        if a.dtype.kind in 'fc' and not np.isfinite(a).all():
            raise ValueError(f'Nonfinite array: {key}')
        if key.endswith('_json'):
            if a.ndim != 0 or a.dtype.kind not in 'US':
                raise ValueError(f'{key} must be scalar JSON text')
            json.loads(a.item())
    t, c, f, m = (np.asarray(data[k]) for k in ('time','coefficients','fields','modes'))
    if t.ndim != 1 or not len(t) or t.dtype.kind not in 'fi' or np.any(np.diff(t) <= 0):
        raise ValueError('time must be finite and strictly increasing')
    if c.ndim != 2 or c.dtype.kind not in 'fi' or f.ndim != 1 or f.dtype.kind not in 'US':
        raise ValueError('Invalid coefficients/fields types or ranks')
    if c.shape != (len(t),len(f)) or m.ndim != 2 or m.shape[0] != len(f) or m.dtype.kind not in 'iu':
        raise ValueError('Inconsistent DOF shapes')
    if m.shape[1] not in (1,2,3) or np.any(m < 0):
        raise ValueError('Invalid mode indices')
    order = [(str(x), tuple(map(int,y))) for x,y in zip(f,m)]
    if len(set(order)) != len(order):
        raise ValueError('Duplicate DOF identifiers')
    if 'order_json' in data and decode(data,'order_json') != [[x,list(y)] for x,y in order]:
        raise ValueError('order_json disagrees with DOF arrays')
    if 'initial_coefficients' in data:
        if data['initial_coefficients'].shape != (len(f),) or not np.array_equal(data['initial_coefficients'],c[0]):
            raise ValueError('Initial coefficients disagree with first snapshot')
    if 'archive_schema_version' in data and str(data['archive_schema_version'].item()).split('.')[0] != '1':
        raise ValueError('Unsupported archive major version')
    meta = decode(data,'metadata_json')
    if not isinstance(meta, dict):
        raise ValueError('metadata_json must contain a JSON object')
    representation = meta.get('representation','cosine_galerkin')
    if representation not in ('cosine_galerkin','cell_average','tensor_galerkin'):
        raise ValueError('Unknown representation')
    if representation == 'tensor_galerkin' and 'basis_json' not in data:
        raise ValueError('Tensor Galerkin archive requires basis_json')
    if 'basis_json' in data:
        b = decode(data, 'basis_json')
        required = {'kinds','lengths','grid_shape','effective_modes'}
        if not isinstance(b, dict) or required - b.keys():
            raise ValueError('basis_json requires kinds, lengths, grid_shape, effective_modes')
        if any(len(b[key]) != m.shape[1] for key in ('kinds','lengths','effective_modes')):
            raise ValueError('Basis dimension mismatch')
        if any(k not in ('sine','cosine','fourier') for k in b['kinds']):
            raise ValueError('Unknown tensor basis')
        import itertools
        if len(b['grid_shape']) != m.shape[1] or any(n < 2 for n in b['grid_shape']) or any(not np.isfinite(L) or L<=0 for L in b['lengths']):
            raise ValueError('Invalid basis grid')
        indices=[]
        for kind,effective in zip(b['kinds'],b['effective_modes']):
            if type(effective) is not int or effective < 1: raise ValueError('Invalid effective modes')
            indices.append(range(2*effective+1) if kind=='fourier' else
                           (range(1,effective+1) if kind=='sine' else range(effective)))
        expected=np.array(list(itertools.product(*indices)),dtype=np.int64)
        for name in dict.fromkeys(f):
            if not np.array_equal(m[f==name],expected):raise ValueError('Spectral DOF ordering mismatch')

    if representation == 'cell_average':
        grid = decode(data,'grid_json'); shape = tuple(grid['shape'])
        if len(shape) != m.shape[1] or any(n < 2 for n in shape):
            raise ValueError('Invalid grid shape')
        expected = np.array(list(np.ndindex(shape)))
        for name in dict.fromkeys(f):
            if not np.array_equal(m[f == name],expected):
                raise ValueError('Grid DOF order must be C order')
        registry = decode(data,'field_metadata_json')
        if [r['id'] for r in registry] != list(dict.fromkeys(f)):
            raise ValueError('Registry and DOFs disagree')
        by_id = {r['id']: r for r in registry}
        for r in registry:
            if not np.isfinite(r['birth_time']) or r['birth_time'] < t[0]:
                raise ValueError('Invalid field birth time')
            if r['parent_id'] is not None and r['parent_id'] not in by_id:
                raise ValueError('Missing genealogy parent')
            death = r['death_time']
            if death is not None and (not np.isfinite(death) or death < r['birth_time']):
                raise ValueError('Invalid field death time')
            if death is not None and np.any(c[t >= death][:, f == r['id']] != 0):
                raise ValueError('Nonzero retired field')
            mask = t < r['birth_time']
            if np.any(c[mask][:,f == r['id']] != 0):
                raise ValueError('Nonzero field before birth')
    if representation == 'cell_average':
        nf = len(registry)
        for key in ('observables_mass', 'observables_active', 'observables_components', 'observables_components_checked_at'):
            if key in data and data[key].shape != (len(t), nf):
                raise ValueError(f'Invalid observable shape: {key}')
    if 'events_utf8' in data:
        raw, offsets = data['events_utf8'], data['events_offsets']
        if raw.dtype != np.uint8 or raw.ndim != 1 or offsets.ndim != 1 or offsets.dtype.kind not in 'iu' or len(offsets)<1 or offsets[0]!=0 or offsets[-1]!=len(raw) or np.any(np.diff(offsets)<=0):
            raise ValueError('Invalid indexed event log')
        if data['events_time'].shape != (len(offsets)-1,):
            raise ValueError('Invalid event times')
    return data


def write_archive(path, data):
    """Validate then atomically replace one self-contained NPZ (no pickle)."""
    payload = {k:np.asarray(v) for k,v in data.items()}
    payload.setdefault('archive_schema_version',np.array(SCHEMA_VERSION))
    validate(payload)
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent,suffix='.npz')
    try:
        with os.fdopen(fd,'wb') as stream:
            np.savez_compressed(stream,**payload)
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


# Load NPZ with allow_pickle=False, copy arrays out of the file, then validate the archive
# contract. Returns the array dictionary, not reconstructed fields.
def read_archive(path):
    with np.load(path,allow_pickle=False) as source:
        data = {k:source[k] for k in source.files}
    return validate(data)


def reconstruct_grid(data, index):
    """Exact archived cell averages, shape (registered fields, *grid.shape)."""
    if decode(data,'metadata_json').get('representation') != 'cell_average':
        raise ValueError('Use cosine reconstruction for a Galerkin archive')
    shape = tuple(decode(data,'grid_json')['shape'])
    return data['coefficients'][index].reshape((-1,)+shape)


def pack_events(events, json_limit=1000):
    """Small logs stay JSON; large logs use indexed UTF-8 records in numeric arrays.

    Offsets permit per-event decoding without parsing one giant JSON document.
    NPZ compression keeps a single portable file; no external sidecars required.
    """
    if len(events)<=json_limit: return {'events_json':json_array(events)}
    chunks=[json.dumps(e,ensure_ascii=False,allow_nan=False).encode('utf-8') for e in events]
    offsets=np.concatenate(([0],np.cumsum([len(c) for c in chunks],dtype=np.int64)))
    return {'events_utf8':np.frombuffer(b''.join(chunks),dtype=np.uint8),
            'events_offsets':offsets,'events_time':np.array([e['time'] for e in events]),
            'events_encoding':np.array('indexed-utf8-json-v1')}


# Yield decoded event records from packed archive metadata; event order is preserved for
# replay/audit.
def iter_events(data):
    if 'events_json' in data:
        yield from decode(data,'events_json')
    elif 'events_utf8' in data:
        raw=data['events_utf8']; offsets=data['events_offsets']
        for a,b in zip(offsets[:-1],offsets[1:]):
            yield json.loads(raw[a:b].tobytes().decode('utf-8'))
