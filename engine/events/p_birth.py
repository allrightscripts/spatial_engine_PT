"""One transactional physical event map for manual AND stochastic P births.

Generic policies operate on scalar dictionaries, never reaction formulas. All
random draws use sim.rng. Bounds use explicit clipping; invalid Onsager matrices
fail loudly (no hidden projection changing the old block). Preparation validates
all parameters/mobility before appending a field to the registry.
"""
from copy import deepcopy
import numpy as np
from ..model.mobility import validate_onsager
from ..model.birth_profiles import evaluate_birth_profile
from .birth_source import select_source
from .provenance import enabled, next_field_id, set_P


# Validate a finite ordered pair [low, high] and return float endpoints; used by parameter
# and mobility policies.
def bounds(pair):
    a = np.asarray(pair, float)
    if a.shape != (2,) or not np.isfinite(a).all() or a[0] > a[1]:
        raise ValueError('Bounds must be finite [low, high]')
    return float(a[0]), float(a[1])


# Apply additive Gaussian mutation using the supplied generator. Sigma=0 returns the value
# without consuming a random draw.
def mutate_value(value, sigma, rng):
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError('Mutation sigma must be finite and nonnegative')
    return float(value + (rng.normal(0, sigma) if sigma else 0))


# Return the registry index of an active P inheritance parent, or None if none exists.
# Explicit parent_id overrides selection; zero total mass falls back to uniform selection.
def select_parent(sim, config):
    ids = [i for i, f in enumerate(sim.registry) if f['active'] and f['family'] == 'P']
    explicit = config.get('parent_id')
    if explicit is not None:
        i = sim.index(explicit)
        if i not in ids:
            raise ValueError('Birth parent must be an active P')
        return i
    if not ids:
        return None
    policy = config.get('inheritance_parent_policy', config.get('parent_policy', 'random_existing_P'))
    if policy not in ('random_existing_P', 'weighted_by_mass'):
        raise ValueError('Unknown parent policy')
    weights = None
    if policy == 'weighted_by_mass':
        masses = np.maximum(sim.u[ids].reshape(len(ids), -1).sum(axis=1) * sim.grid.volume, 0)
        if masses.sum() > 0:
            weights = masses / masses.sum()
    return int(sim.rng.choice(ids, p=weights))


# Build scalar parameters by manual, random or inherit/mutate policy and explicit
# fallback; clip configured ranges and return values plus audit metadata.
def create_parameters(sim, c, parent):
    policy = c.get('parameter_policy', c.get('policy', 'manual'))
    defaults = deepcopy(sim.config.get('family_defaults', {}).get('P', {}))
    defaults.update(c.get('field_parameters', {}))
    ranges = {k: bounds(v) for k, v in c.get('parameter_ranges', {}).items()}
    effective = policy
    if policy == 'inherit_mutate' and parent is None:
        fallback = c.get('fallback', 'family_default')
        if fallback not in ('family_default', 'random'):
            raise ValueError('Fallback must be family_default or random')
        effective = 'manual' if fallback == 'family_default' else 'random'
    if effective == 'manual':
        values = defaults
    elif effective == 'random':
        values = {**defaults, **{k: float(sim.rng.uniform(*r)) for k, r in ranges.items()}}
    elif effective == 'inherit_mutate':
        values = {k: mutate_value(v, float(c.get('mutation_sigma', 0)), sim.rng)
                  for k, v in sim.registry[parent]['parameters'].items() if k != 'diffusion'}
        # Diffusion has its own independent policy below.
        values['diffusion'] = sim.registry[parent]['parameters'].get('diffusion', 0)
    else:
        raise ValueError('Unknown parameter policy')
    raw = deepcopy(values)
    if c.get('bounds_policy', 'clip') != 'clip':
        raise ValueError('Supported parameter bounds_policy: clip')
    for k, r in ranges.items():
        if k not in values:
            raise ValueError(f'Missing manual/default parameter {k}')
        values[k] = float(np.clip(values[k], *r))
    sim.check_parameters(values)
    return values, {'requested': policy, 'effective': effective, 'bounds_policy': 'clip', 'before_bounds': raw}


# Prepare an enlarged field mobility matrix while preserving its old block; validate
# symmetry/PSD when requested. Return matrix and audit before appending the child.
def expand_mobility(sim, c, fields, parent, parameters, t):
    mobility = sim.model.mobility
    if not hasattr(mobility, 'birth_matrix') or not hasattr(mobility, 'assign_birth_matrix'):
        raise TypeError('Model mobility must implement birth_matrix/assign_birth_matrix for P birth')
    old = np.asarray(mobility.birth_matrix(sim.u[[sim.index(f['id']) for f in fields]], fields, sim.grid, t), float)
    n = len(fields)
    if old.shape != (n, n):
        raise ValueError('Model birth_matrix shape mismatch')
    policy = c.get('mobility_policy', 'manual')
    effective = policy
    if policy == 'inherit_mutate' and parent is None:
        effective = 'random' if c.get('fallback', 'family_default') == 'random' else 'manual'
    diffusion_bounds = bounds(c['diffusion_range']) if 'diffusion_range' in c else None
    coupling_bounds = bounds(c.get('coupling_range', [0, 0]))
    cross = bool(c.get('cross_couplings', False))
    if effective == 'manual':
        D = float(c.get('diffusion', parameters.get('diffusion', .01)))
        couplings = c.get('couplings', {})
        if set(couplings) - {f['id'] for f in fields}:
            raise ValueError('Unknown field in manual couplings')
        row = np.array([float(couplings.get(f['id'], 0)) for f in fields])
    elif effective == 'random':
        if diffusion_bounds is None:
            raise ValueError('Random mobility needs diffusion_range')
        D = float(sim.rng.uniform(*diffusion_bounds))
        row = sim.rng.uniform(*coupling_bounds, size=n) if cross else np.zeros(n)
    elif effective == 'inherit_mutate':
        p = next(i for i, f in enumerate(fields) if f['id'] == sim.registry[parent]['id'])
        sigma = float(c.get('mobility_mutation_sigma', 0))
        D = mutate_value(old[p, p], sigma, sim.rng)
        row = np.array([mutate_value(v, sigma, sim.rng) for v in old[p]]) if cross else np.zeros(n)
        if cross and 'coupling_range' in c:
            row = np.clip(row, *coupling_bounds)
    else:
        raise ValueError('Unknown mobility policy')
    raw_D = D
    if diffusion_bounds is not None:
        D = float(np.clip(D, *diffusion_bounds))
    parameters['diffusion'] = D
    sim.check_parameters(parameters)
    new = np.zeros((n + 1, n + 1)); new[:n, :n] = old
    new[-1, :n] = row; new[:n, -1] = row; new[-1, -1] = D
    validation = c.get('onsager', {})
    if validation.get('invalid_policy', 'fail') != 'fail':
        raise ValueError('Supported Onsager invalid_policy: fail (old block is preserved)')
    dissipative = validation.get('validate', sim.config.get('mobility_validation', {}).get('onsager', getattr(mobility, 'dissipative_onsager', False)))
    validate_onsager(new, dissipative, float(validation.get('tolerance', 1e-10)))
    return new, {'policy': policy, 'effective': effective, 'matrix': new.tolist(),
                 'field_order': [f['id'] for f in fields], 'old_block_preserved': True,
                 'diffusion_before_bounds': raw_D, 'diffusion': D,
                 'onsager_validated': bool(dissipative), 'invalid_policy': 'fail'}


# Event-map order matters: select spatial source, construct physical profile,
# then independently choose inheritance and create parameters/mobility.
# Use sim.rng for event randomness; never introduce global np.random draws.
# The next segment projects the newborn onto the existing Galerkin basis.
# Select spatial source and construct its profile, then independently generate
# inherited/random parameters and mobility. Append one global field and enrich event
# metadata; no integration occurs here.
def p_birth(sim, event):
    c = deepcopy(sim.config.get('stochastic_events', {}).get('p_birth', {}))
    c.update(deepcopy(event.parameters))
    source_index, source_audit = select_source(sim, c, event)
    source_field = sim.registry[source_index] if source_index is not None else None
    profile_U = sim.u if source_index is None else sim.u[[source_index]]
    profile_fields = sim.registry if source_index is None else [source_field]
    newborn, profile_audit = evaluate_birth_profile(
        profile_U, profile_fields, sim.grid, event.time, c, getattr(sim, 'accessibility', None))
    if source_field is not None and c.get('birth_profile') == 'inside_T' and profile_audit['fallback_used'] is not None:
        raise ValueError('per_T inside_T source below epsilon: refusing uniform fallback outside source; reduce epsilon')
    before = deepcopy(sim.rng.bit_generator.state)
    policy = c.get('parameter_policy', c.get('policy', 'manual'))
    needs_parent = policy == 'inherit_mutate' or c.get('mobility_policy') == 'inherit_mutate' or c.get('parent_id') is not None
    parent = select_parent(sim, c) if needs_parent else None
    parameters, parameter_audit = create_parameters(sim, c, parent)
    fields = [f for f in sim.registry if f['active']]
    matrix, mobility_audit = expand_mobility(sim, c, fields, parent, parameters, event.time)
    amplitude = float(profile_audit['birth_profile_parameters']['amplitude'])
    name = c.get('id', next_field_id(sim) if enabled(sim) or c.get('source_mode')=='per_T' else sim.next_name('P'))
    parent_field = sim.registry[parent] if parent is not None else None
    model_type = c.get('model_type', parent_field.get('model_type', 'progressive_v1') if parent_field else 'progressive_v1')
    if hasattr(sim.model.reaction, 'validate_type'):
        sim.model.reaction.validate_type(model_type)
    parent_id = None if parent_field is None else parent_field['id']
    sim.add_field(name, 'P', newborn,
                  event.time, parameters, parent=parent_id, bc=c.get('bc'))
    child = sim.registry[-1]
    child.update(model_type=model_type, birth_segment=getattr(sim, 'birth_segment', 0))
    inherited = parameter_audit['effective']=='inherit_mutate' or mobility_audit['effective']=='inherit_mutate'
    set_P(sim,child,source_field,parent_field if inherited else None)
    sim.model.mobility.assign_birth_matrix(fields + [child], matrix)
    mobility_audit['field_order'].append(name)
    link = {k: child[k] for k in ('parent_id', 'generation', 'birth_time', 'birth_segment')}
    link['child_id'] = name
    sim.genealogy.append(link)
    event.targets = [name]
    event.parameters = {**c, 'child_id': name, **link, 'model_type': model_type,
                        'resulting_parameters': deepcopy(child['parameters']),
                        'parameter_creation': parameter_audit, 'mobility_creation': mobility_audit,
                        'birth_amplitude': amplitude, **profile_audit, **source_audit,
                        'child_display_label':child['display_label'],
                        **{k:deepcopy(child[k]) for k in ('source_T_id','source_T_lineage','source_T_lineage_at_birth',
                             'source_T_display_label','source_birth_ordinal','lineage_path',
                             'inheritance_parent_id','inheritance_parent_lineage','inheritance_parent_display_label')},
                        'initial_state': ('amplitude * accessibility' if profile_audit['birth_profile'] == 'uniform'
                                          else 'birth_profile * accessibility')}
    event.rng.update(seed=sim.config['seed'], state_before_parameters=before,
                     state_after_parameters=deepcopy(sim.rng.bit_generator.state))
