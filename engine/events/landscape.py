"""Persistent accessibility maps; temporary layers compose without undo bugs.

M = base * product(active layer factors). Each event installs a factor
1-strength*mask, sampled at event start; a temporary layer is removed at its
exact end. Overlapping permanent/temporary changes therefore commute. Events
modify the operator, not U. Fully forbidden regions have zero flux and zero
reaction in the continuum PDE; existing values there are frozen, not erased.
"""
from copy import deepcopy
import numpy as np
from ..model.masks import evaluate_mask


# Install/remove an accessibility layer and recompute the product of active layers. Alters
# transport/reaction accessibility, not existing field values.
def landscape(sim, event):
    p = event.parameters
    if p.get('phase') == 'end':
        sim.landscape_layers = [layer for layer in sim.landscape_layers
                                if layer['event_id'] != p['start_event_id']]
    else:
        duration = float(p.get('duration', 0))
        if not np.isfinite(duration) or duration < 0 or not 0 <= event.strength <= 1:
            raise ValueError('Landscape needs duration >= 0 and strength in [0,1]')
        mask = evaluate_mask(sim.grid, event.geometry, event.time)
        layer = {'event_id': f'event_{len(sim.events):08d}',
                 'end': event.time + duration if duration else None,
                 'factor': 1 - event.strength * mask}
        sim.landscape_layers.append(layer)
        p.update(effect_type='permeability_factor', mask_sample_time=event.time,
                 transport='div(M L grad(mu))', reaction='M * R',
                 state_effect='none; existing forbidden-region values are not erased')
    sim.accessibility = sim.landscape_base.copy()
    for layer in sim.landscape_layers:
        sim.accessibility *= layer['factor']
    p['accessibility_min'] = float(sim.accessibility.min())
    p['accessibility_max'] = float(sim.accessibility.max())
