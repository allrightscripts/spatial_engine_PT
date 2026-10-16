"""CHANGE EVENT GEOMETRY HERE. JSON selects names; never evaluates code.

Register callables accepting (*grid.mesh, t, params) and returning a scalar or
array broadcastable to grid.shape in [0,1]. An event samples its mask ONCE at
its start time; temporary events keep that frozen mask for their lifetime.
Built-ins are shared by initial conditions, catastrophe and landscape events.
"""
import numpy as np

# EXTENSION POINT: @register_mask("name") makes geometry available by name.
# Use physical coordinates from grid.mesh, not integer pixel positions.
# For 1D/2D/3D support, allow absent Y/Z arguments as in the example below.
MASKS = {}


# Return a decorator exposing a callable under a JSON-selectable geometry mask name.
# Register rules at module import, not during integration.
def register_mask(name):
    # Store the supplied geometry mask callable under the captured name and return it
    # unchanged; registration itself does not execute the rule.
    def register(function):
        MASKS[name] = function
        return function
    return register


@register_mask('my_catastrophe')
def my_catastrophe(X, Y=None, Z=None, *, t, params):
    """Example custom interval/ellipse/ellipsoid, coefficients supplied by JSON."""
    q = ((X - params.get('a', .5)) / params.get('A', .1)) ** 2
    if Y is not None:
        q = q + ((Y - params.get('b', .5)) / params.get('B', .2)) ** 2
    if Z is not None:
        q = q + ((Z - params.get("c", .5)) / params.get("C", .2)) ** 2
    return q <= 1


def _builtin_mask(grid, geometry):
    """Masks sampled at cell centres; supports smooth and hard local geometry."""
    g=geometry; kind=g.get('type','all'); d=len(grid.shape)
    if kind=='all': return np.ones(grid.shape)
    center=np.asarray(g.get('center',[L/2 for L in grid.lengths]),float)
    if center.shape!=(d,) or not np.isfinite(center).all(): raise ValueError('Invalid mask center')
    if kind in ('interval','rectangle','box','cuboid'):
        low=np.asarray(g['low'],float); high=np.asarray(g['high'],float)
        if low.shape!=(d,) or high.shape!=(d,) or np.any(high<low): raise ValueError('Invalid rectangle')
        return np.logical_and.reduce([(x>=a)&(x<=b) for x,a,b in zip(grid.mesh,low,high)]).astype(float)
    if kind in ('circle','ellipse','sphere','ellipsoid','gaussian'):
        radii=np.broadcast_to(np.asarray(g.get('radii',g.get('radius',.1)),float),(d,))
        if np.any(radii<=0) or not np.isfinite(radii).all(): raise ValueError('Mask radii must be positive')
        q=sum(((x-c)/r)**2 for x,c,r in zip(grid.mesh,center,radii))
        return np.exp(-.5*q) if kind=='gaussian' else (q<=1).astype(float)
    raise ValueError(f'Unknown mask {kind}')



# Evaluate a named custom mask or built-in geometry at time t, broadcast to grid.shape and
# require finite values in [0,1].
def evaluate_mask(grid, geometry, t=0.0):
    name = geometry.get('mask', geometry.get('type', 'all'))
    if name in MASKS:
        value = MASKS[name](*grid.mesh, t=float(t), params=geometry.get('parameters', {}))
    else:
        value = _builtin_mask(grid, {**geometry, 'type': name})
    mask = np.broadcast_to(np.asarray(value, float), grid.shape).copy()
    if not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
        raise ValueError('Mask must be finite and in [0,1]')
    return mask
