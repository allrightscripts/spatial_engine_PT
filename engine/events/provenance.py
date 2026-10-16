"""Additive provenance metadata. Stable IDs never derive from display labels in v0.9 mode."""
from copy import deepcopy


# Enable v0.9 naming/order when lineage.enabled or per_T birth is configured; old configs
# retain legacy generated IDs/order.
def enabled(sim):
    return bool(sim.config.get('lineage', {}).get('enabled', False) or
                sim.config.get('stochastic_events', {}).get('p_birth', {}).get('source_mode') == 'per_T')


# Find an unused stable field_NNNNNN ID without deriving it from a display label.
# Historical retired IDs remain reserved.
def next_field_id(sim):
    taken = {f['id'] for f in sim.registry}
    n = len(sim.registry)
    while f'field_{n:06d}' in taken:
        n += 1
    return f'field_{n:06d}'


# Add provenance defaults to a newly appended registry row; root T labels use distinct
# namespaces when multiple roots exist.
def initialize(sim, child, parent=None):
    child.update(lineage_path=[], lineage_root_id=child['id'],
                 display_label=child['id'], split_parent_id=None,
                 source_T_id=None, source_T_lineage=None, source_T_lineage_at_birth=None,
                 inheritance_parent_id=None, inheritance_parent_lineage=None,
                 source_birth_ordinal=None, birth_event_id=None)
    if child['family'] == 'T':
        roots = [f for f in sim.registry[:-1] if f['family']=='T' and f.get('split_parent_id') is None]
        child['lineage_root_label'] = 'T' if not roots else 'T@'+child['id']
        child['display_label'] = child['lineage_root_label']


# Set the explicit split parent, inherited root namespace and parent path plus child
# index. Updates metadata only, not field values.
def set_split(child, parent, branch):
    child['split_parent_id'] = parent['id']
    child['lineage_root_id'] = parent.get('lineage_root_id', parent['id'])
    child['lineage_root_label'] = parent.get('lineage_root_label', parent['family'])
    child['lineage_path'] = list(parent.get('lineage_path', [])) + [int(branch)]
    child['display_label'] = child['lineage_root_label']+'_'+'.'.join(map(str, child['lineage_path']))


# source is the historical spatial T parent; inheritance is a separate P.
# Never infer either relation from display_label, which is presentation only.
# Assign a per-source historical birth ordinal, source lineage and separate inheritance
# relation. Retired siblings still count, so ordinals are not reused.
def set_P(sim, child, source, inheritance):
    # Source birth ordinals are independent of inheritance and survive extinction.
    prior = [f.get('source_birth_ordinal') for f in sim.registry[:-1]
             if f['family']=='P' and f.get('source_T_id') == (source['id'] if source else None)]
    ordinal = max((n for n in prior if n is not None), default=-1)+1
    path = deepcopy(source['lineage_path']) if source else None
    tag = '.'.join(map(str,path)) if path else ('root' if source else 'global')
    if source and source.get('lineage_root_label','T') != 'T':
        tag = source['lineage_root_id']+':'+tag
    child.update(source_T_id=source['id'] if source else None,
                 source_T_lineage=path, source_T_lineage_at_birth=deepcopy(path),
                 source_T_display_label=source['display_label'] if source else None,
                 source_birth_ordinal=ordinal, lineage_path=[ordinal],
                 lineage_root_id=source['id'] if source else None,
                 display_label=f'P_{tag};{ordinal}',
                 inheritance_parent_id=inheritance['id'] if inheritance else None,
                 inheritance_parent_lineage=deepcopy(inheritance['lineage_path']) if inheritance else None,
                 inheritance_parent_display_label=inheritance['display_label'] if inheritance else None)


# Derive typed T_split, P_source_birth and P_parameter_inheritance edges from explicit
# IDs, including creation time/event ID.
def edges(registry):
    result=[]
    for f in registry:
        for key,kind in [('split_parent_id','T_split'),('source_T_id','P_source_birth'),
                         ('inheritance_parent_id','P_parameter_inheritance')]:
            if f.get(key) is None or (key=='split_parent_id' and f['family']!='T'):
                continue
            result.append({'edge_type':kind,'parent_id':f[key],'child_id':f['id'],
                           'birth_time':f['birth_time'],'birth_event_id':f.get('birth_event_id')})
    return result


# Attach the recording event ID to its newborn fields. Initial fields have no creating
# event and retain null IDs.
def record_creation(sim, event):
    if event.kind not in ('p_birth','nucleation','split'):
        return
    for field in sim.registry:
        if field['id'] in event.targets and field.get('birth_event_id') is None and field['birth_time']==event.time:
            field['birth_event_id']=event.event_id
    event.parameters['birth_event_id']=event.event_id
