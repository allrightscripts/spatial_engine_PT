"""Semantic target selection over stable IDs; inheritance is not source provenance."""
from copy import deepcopy


# Resolve source-provenance subtrees, not parameter-inheritance descendants.
# An inactive root may still have active descendants. Resolution returns the
# actual registry objects; event recording stores stable IDs and display labels.
# Return active registry rows matching all/family/source-subtree semantics. Subtrees
# traverse split/source links, never P parameter inheritance.
def resolve_selector(registry, selector):
    if not isinstance(selector, dict):
        raise ValueError('target_selector must be an object')
    mode=selector.get('mode'); active=[f for f in registry if f['active']]
    if mode=='all_active':return active
    if mode=='family':
        families=selector.get('families')
        if not isinstance(families,list) or not families or any(not isinstance(f,str) for f in families):
            raise ValueError('family selector needs nonempty families list')
        return [f for f in active if f['family'] in families]
    if mode=='lineage_subtree':
        root=selector.get('root_field_id');known={f['id'] for f in registry}
        if root not in known:raise ValueError('Unknown lineage root_field_id')
        include_root=selector.get('include_root',True);descend=selector.get('include_descendants',True)
        if type(include_root) is not bool or type(descend) is not bool:raise ValueError('Subtree flags must be booleans')
        reached={root}
        if descend:
            while True:
                new={f['id'] for f in registry if f.get('split_parent_id') in reached or f.get('source_T_id') in reached}
                if new<=reached:break
                reached |= new
        if not include_root:reached.discard(root)
        return [f for f in active if f['id'] in reached]
    raise ValueError(f'Unknown target_selector mode: {mode}')


# Resolve semantic targets at application time and store their stable IDs/labels.
# Duration-end events keep start-time resolution; mixed selector/exact targets fail.
def resolve_event(sim,event):
    p=event.parameters
    if p.get('phase')=='end':return
    selector=p.get('target_selector')
    if selector is not None:
        if event.kind not in ('catastrophe','sabotage','manual','mutation','extinction'):
            raise ValueError(f'target_selector not supported for {event.kind}')
        if event.targets or p.get('target_families'):
            raise ValueError('Use either target_selector or targets/target_families, not both')
        selected=resolve_selector(sim.registry,selector)
        event.targets=[f['id'] for f in selected]
        p['requested_selector']=deepcopy(selector)
    else:
        selected=[sim.registry[i] for i in sim.target_indices(event.targets)] if event.targets else []
        # Birth targets name the newborn and are set by its handler.
        if event.kind in ('p_birth','nucleation','split','landscape'):return
        p['requested_selector']={'mode':'exact_ids','targets':list(event.targets)}
    p['resolved_target_ids']=[f['id'] for f in selected]
    p['resolved_target_labels']=[f.get('display_label',f['id']) for f in selected]
    if selector is not None:p['target_resolution_policy']='snapshot_at_event_start'
