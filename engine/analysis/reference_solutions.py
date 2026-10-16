"""Reusable real-field references; no imports of simulation or model code.

Register a deterministic callable (grid, t, params) returning field samples.
The cosine benchmark accepts one tensor mode, or a finite smooth cosine sum.
"""
import numpy as np

# EXTENSION POINT: @register_reference("name") adds an analytic/reference field.
# Sample on the supplied grid, using its physical lengths; return grid.shape.
# The analyzer compares this to every requested saved field/frame. It does not
# solve the PDE or infer the reference parameters from a cultural run.
REFERENCES = {}


# Return a decorator exposing a callable under a JSON-selectable reference solution name.
# Register rules at module import, not during integration.
def register_reference(name):
    # Store the supplied reference solution callable under the captured name and return it
    # unchanged; registration itself does not execute the rule.
    def register(function):
        REFERENCES[name] = function
        return function
    return register


# Convert a reference parameter to float and reject NaN/Inf with its parameter name in the
# error.
def _finite(value, name):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


# Evaluate one cosine product or a sum of products on grid.mesh, with analytic diffusion
# decay and optional constant offset. Validate axis counts and nonnegative integer wave
# numbers.
@register_reference('diffusion_neumann_cosine')
def diffusion_neumann_cosine(grid, t, params):
    D = _finite(params.get('D', .1), 'D')
    t0 = _finite(params.get('t0', 0.0), 't0')
    t = _finite(t, 't')
    if D < 0 or t < t0:
        raise ValueError('Diffusion reference needs D >= 0 and t >= t0')
    dim = len(grid.shape)
    offset = _finite(params.get('offset', 0.0), 'offset')
    terms = params.get('terms')
    if terms is None:
        terms = [{'k': params.get('k', [1]*dim), 'amplitude': params.get('amplitude', 1.0)}]
    if not isinstance(terms, list) or not terms:
        raise ValueError('Reference terms must be a nonempty list')
    result = np.full(grid.shape, offset)
    for term in terms:
        if not isinstance(term, dict):
            raise ValueError('Each reference term must be an object')
        k = np.asarray(term.get('k', [1]*dim), float)
        if k.ndim == 0 and dim == 1:
            k = k.reshape(1)
        if k.shape != (dim,) or not np.isfinite(k).all() or np.any(k < 0) or np.any(k != np.floor(k)):
            raise ValueError('Reference k must contain one nonnegative integer per axis')
        amplitude = _finite(term.get('amplitude', 1.0), 'amplitude')
        eigenvalue = np.pi**2 * np.sum((k / np.asarray(grid.lengths))**2)
        value = amplitude * np.exp(-D * eigenvalue * (t-t0))
        for wave, x, length in zip(k, grid.mesh, grid.lengths):
            value = value * np.cos(wave * np.pi * x / length)
        result += value
    return result


# Dispatch by registered reference name and return validated real grid samples for the
# given time/parameters.
def reference_solution(name, grid, t, params=None):
    if name not in REFERENCES:
        raise ValueError(f'Unknown reference solution: {name}')
    params = {} if params is None else params
    if not isinstance(params, dict):
        raise ValueError('Reference parameters must be an object')
    raw = REFERENCES[name](grid, float(t), params)
    if np.iscomplexobj(raw):
        raise ValueError('Reference solution must be real')
    values = np.broadcast_to(np.asarray(raw, float), grid.shape).copy()
    if not np.isfinite(values).all():
        raise ValueError('Reference solution must be finite')
    return values
