"""Select source T at birth, before independent parameter inheritance draws."""
from copy import deepcopy
import numpy as np
from ..model.event_rules import source_hazards


# Return selected T registry index plus scalar hazard/probability audit. Legacy global
# mode returns no concrete source and consumes no source-selection draw.
def select_source(sim, config, event):
    mode=config.get('source_mode','global_T_sum')
    if mode=='global_T_sum':
        return None, {'source_mode':mode,'source_hazard':None,'total_hazard':event.rng.get('instantaneous_rate'),
                      'source_selection_probability':None,'source_selection':'legacy_global_T_sum'}
    if mode!='per_T':raise ValueError(f'Unknown source_mode: {mode}')
    indices,rates=source_hazards(config,sim.u,sim.registry,sim.grid,event.time)
    total=float(rates.sum())
    if total<=0:raise ValueError('per_T birth has no positive source hazard at event application')
    probabilities=rates/total
    requested=config.get('source_T_id')
    before=deepcopy(sim.rng.bit_generator.state)
    if requested is not None:
        if event.source=='integrated_hazard':raise ValueError('source_T_id override is for manual births only')
        matches=[j for j,i in enumerate(indices) if sim.registry[i]['id']==requested]
        if not matches or rates[matches[0]]<=0:raise ValueError('Requested T source must be active with positive hazard')
        j=matches[0];selection='explicit_manual'
    else:
        j=int(sim.rng.choice(len(indices),p=probabilities));selection='competing_hazards'
    source=sim.registry[indices[j]]
    event.rng.update(source_selection_state_before=before,source_selection_state_after=deepcopy(sim.rng.bit_generator.state))
    return indices[j], {'source_mode':mode,'source_hazard':float(rates[j]),'total_hazard':total,
                         'source_selection_probability':float(probabilities[j]),'source_selection':selection}
