"""Deterministic spatial initialization; independent of hazard and RNG.

Register a callable (U, fields, grid, t, params) returning
(values, effective_parameters, fallback_used). Values precede the common
accessibility multiplier and the existing Galerkin projection.
"""
from copy import deepcopy
import numpy as np

# EXTENSION POINT: register a named profile; JSON stores its name and numbers.
# In per_T mode the caller passes ONLY the chosen T field, not the full registry.
# Return finite, nonnegative samples BEFORE applying landscape or projection.
# The shared wrapper below applies accessibility once and writes the audit data.
BIRTH_PROFILES = {}


# Return a decorator exposing a callable under a JSON-selectable newborn profile name.
# Register rules at module import, not during integration.
def register_birth_profile(name):
    # Store the supplied newborn profile callable under the captured name and return it
    # unchanged; registration itself does not execute the rule.
    def register(function):
        BIRTH_PROFILES[name] = function
        return function
    return register


# Sum active T fields among the supplied aligned arrays/metadata. In per_T mode the caller
# supplies only the selected T.
def total_T(U, fields, grid):
    ids = [i for i, f in enumerate(fields)
           if f.get('active', True) and f['family'] == 'T']
    return np.sum(U[ids], axis=0) if ids else np.zeros(grid.shape)


# Convert amplitude to float and require a finite nonnegative value before creating a
# newborn profile.
def _amplitude(params):
    amplitude = float(params['amplitude'])
    if not np.isfinite(amplitude) or amplitude < 0:
        raise ValueError('Birth profile amplitude must be finite and nonnegative')
    return amplitude


# Return a constant profile plus effective parameter audit; the shared wrapper applies
# accessibility afterward.
@register_birth_profile('uniform')
def uniform(U, fields, grid, t, params):
    p = deepcopy(params); p['amplitude'] = _amplitude(p)
    return np.full(grid.shape, p['amplitude']), p, None


# Normalize clipped T by its maximum plus epsilon and raise to power, handling the
# configured zero-T fallback. Does not mutate T or apply the landscape mask.
@register_birth_profile('inside_T')
def inside_T(U, fields, grid, t, params):
    p = deepcopy(params)
    p.update(amplitude=_amplitude(p), power=float(p.get('power', 3.0)),
             epsilon=float(p.get('epsilon', 1e-12)), fallback=p.get('fallback', 'uniform'))
    if not np.isfinite(p['power']) or p['power'] <= 0:
        raise ValueError('Birth profile power must be finite and positive')
    if not np.isfinite(p['epsilon']) or p['epsilon'] <= 0:
        raise ValueError('Birth profile epsilon must be finite and positive')
    if p['fallback'] not in ('uniform', 'fail'):
        raise ValueError('Birth profile fallback must be uniform or fail')
    # Spectral reconstruction can be slightly negative. Clip only the total
    # used to initialize this newborn; never modify an existing T field.
    T = np.maximum(total_T(U, fields, grid), 0.0)
    maximum = float(T.max())
    if maximum <= p['epsilon']:
        if p['fallback'] == 'fail':
            raise ValueError('inside_T birth: T_total_max <= epsilon (fallback=fail)')
        return np.full(grid.shape, p['amplitude']), p, 'uniform'
    # Stable equivalent of T/(maximum+epsilon), including near float limits.
    ratio = (T / maximum) / (1.0 + p['epsilon'] / maximum)
    return p['amplitude'] * ratio ** p['power'], p, None


def evaluate_birth_profile(U, fields, grid, t, config, accessibility=None):
    """Return physical newborn and scalar/config audit; never consume RNG."""
    name = config.get('birth_profile', 'uniform')
    if name not in BIRTH_PROFILES:
        raise ValueError(f'Unknown birth_profile: {name}')
    params = deepcopy(config.get('birth_profile_parameters', {}))
    if not isinstance(params, dict):
        raise ValueError('birth_profile_parameters must be an object')
    params.setdefault('amplitude', config.get('birth_amplitude', 10.0))
    U = np.asarray(U, float)
    if U.shape != (len(fields),) + grid.shape or not np.isfinite(U).all():
        raise ValueError('Invalid state for birth profile')
    T = total_T(U, fields, grid)
    if not np.isfinite(T).all():
        raise ValueError('Nonfinite T total for birth profile')
    values, effective, fallback = BIRTH_PROFILES[name](U, fields, grid, t, params)
    values = np.broadcast_to(np.asarray(values, float), grid.shape).copy()
    M = np.broadcast_to(np.asarray(1.0 if accessibility is None else accessibility, float), grid.shape)
    if not np.isfinite(M).all() or np.any((M < 0) | (M > 1)):
        raise ValueError('Birth accessibility must be finite in [0,1]')
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Birth profile must be finite and nonnegative')
    audit = {'birth_profile': name, 'birth_profile_parameters': effective,
             'T_total_max': float(T.max()), 'fallback_used': fallback}
    if name == 'inside_T':
        audit.update(negative_T_policy='clip_total_for_profile',
                     T_total_min=float(T.min()))
    return values * M, audit
