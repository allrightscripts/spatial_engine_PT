"""CHANGE STOCHASTIC BIRTH LAW HERE (Python formulas; JSON numbers).

A rule returns a finite nonnegative GLOBAL intensity, in births / unit time.
It must be deterministic and side-effect free: adaptive ODE solvers evaluate it
at trial states/times. Never draw random numbers inside a rate function.
Register another function with @register_rate('name'), then select it in JSON.
"""
import numpy as np

RATES = {}


# Return a decorator exposing a callable under a JSON-selectable legacy global rate name.
# Register rules at module import, not during integration.
def register_rate(name):
    # Store the supplied legacy global rate callable under the captured name and return it
    # unchanged; registration itself does not execute the rule.
    def register(function):
        RATES[name] = function
        return function
    return register


@register_rate('default_p_birth_rate')
def p_birth_rate(U, fields, grid, t, params):
    """EXAMPLE only: lambda0 + alpha * integral(total P). Replace freely.

    Family metadata, not array position, identifies P. Spectral roundoff may
    produce tiny negative masses; this example counts nonnegative total mass.
    """
    ids = [i for i, f in enumerate(fields) if f['family'] == 'P']
    mass = float(U[ids].sum() * grid.volume) if ids else 0.0
    return params.get('lambda0', 0.0) + params.get('alpha', 0.0) * max(mass, 0.0)


# Evaluate a named legacy global intensity and reject negative/nonfinite values. Units are
# births per model-time unit, not probability per step.
def evaluate_rate(name, U, fields, grid, t, params):
    if name not in RATES:
        raise ValueError(f'Unknown P birth rate: {name}; register it in event_rules.py')
    rate = float(RATES[name](U, fields, grid, t, params))
    if not np.isfinite(rate) or rate < 0:
        raise ValueError('P birth intensity must be finite and nonnegative')
    return rate

# Per-source rules are separate from the legacy global registry.
# EXTENSION POINT: add @register_source_rate("your_name") below, then select
# it with stochastic_events.p_birth.source_rate_function. Return births/time,
# not a probability or an already-integrated count. The clock performs integration.
# Keep this callback pure: adaptive solvers may call it repeatedly at trial states.
SOURCE_RATES = {}


# Return a decorator exposing a callable under a JSON-selectable per-T source rate name.
# Register rules at module import, not during integration.
def register_source_rate(name):
    # Store the supplied per-T source rate callable under the captured name and return it
    # unchanged; registration itself does not execute the rule.
    def register(function):
        SOURCE_RATES[name] = function
        return function
    return register


@register_source_rate('default_p_birth_rate_for_source')
def p_birth_rate_for_source(source_index, U, fields, grid, t, params):
    """Example: lambda0 * positive T mass + alpha * associated positive P mass.

    A zero-mass T is ineligible. U and metadata are aligned, including when
    called from a fixed-dimensional provider. This is user-editable model code.
    """
    mass=float(np.maximum(U[source_index],0).sum()*grid.volume)
    if mass<=0:return 0.0
    source=fields[source_index]['id']
    ids=[i for i,f in enumerate(fields) if f['family']=='P' and f.get('active',True) and f.get('source_T_id')==source]
    p_mass=float(np.maximum(U[ids],0).sum()*grid.volume) if ids else 0.0
    return params.get('lambda0',0.0)*mass + params.get('alpha',0.0)*p_mass


# Evaluate the registered per-source law for all active T fields; return aligned indices
# and a finite nonnegative rate array.
def source_hazards(config, U, fields, grid, t):
    name=config.get('source_rate_function','default_p_birth_rate_for_source')
    if name not in SOURCE_RATES:raise ValueError(f'Unknown per-T rate: {name}')
    indices=[i for i,f in enumerate(fields) if f['family']=='T' and f.get('active',True)]
    rates=np.array([float(SOURCE_RATES[name](i,U,fields,grid,t,config.get('rate_parameters',{}))) for i in indices])
    if not np.isfinite(rates).all() or np.any(rates<0):raise ValueError('Per-T hazards must be finite and nonnegative')
    if not np.isfinite(rates.sum()):raise ValueError('Nonfinite total per-T hazard')
    return indices,rates


# Dispatch between legacy global_T_sum and the sum of per_T hazards. This controls the
# clock, not newborn spatial initialization.
def evaluate_birth_rate(config, U, fields, grid, t):
    mode=config.get('source_mode','global_T_sum')
    if mode=='per_T':return float(source_hazards(config,U,fields,grid,t)[1].sum())
    if mode!='global_T_sum':raise ValueError(f'Unknown source_mode: {mode}')
    return evaluate_rate(config.get('rate_function','default_p_birth_rate'),U,fields,grid,t,config.get('rate_parameters',{}))
