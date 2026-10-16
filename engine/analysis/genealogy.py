"""Read-only source and inheritance forests. Never infer provenance from names."""
import csv
from copy import deepcopy
import json
from pathlib import Path

COLUMNS=('field_id','display_label','family','lineage_path','source_T_id','source_T_label',
         'inheritance_parent_id','inheritance_parent_label','birth_time','death_time',
         'birth_segment','generation','birth_event_id','source_birth_ordinal')


# Load registry metadata and derive typed source/inheritance edges. Reject duplicate IDs
# or missing parents; legacy unknown sources stay unknown.
def load_genealogy(path):
    manifest=json.loads(Path(path).read_text(encoding='utf-8'))
    registry=deepcopy(manifest['field_registry'])
    by_id={f['id']:f for f in registry}
    if len(by_id)!=len(registry):raise ValueError('Duplicate field IDs in manifest')
    warnings=[]
    old='provenance_schema_version' not in manifest
    if old:warnings.append('Legacy manifest: source T is unknown unless explicitly stored; names are not parsed.')
    for f in registry:
        f.setdefault('display_label',f.get('genealogy_label',f['id']))
        f.setdefault('lineage_path',None)
        parent=by_id.get(f.get('parent_id'))
        # v0.8 parent_id has known T-split / P-inheritance semantics, but never
        # gives a spatial T-source for a P. Leave the latter unknown.
        f.setdefault('split_parent_id',parent['id'] if old and parent and parent['family']==f['family']=='T' else None)
        f.setdefault('inheritance_parent_id',parent['id'] if old and parent and parent['family']==f['family']=='P' else None)
        f.setdefault('source_T_id',None)
    edges=[]
    for f in registry:
        for key,kind in [('split_parent_id','T_split'),('source_T_id','P_source_birth'),('inheritance_parent_id','P_parameter_inheritance')]:
            parent=f.get(key)
            if parent is None:continue
            if parent not in by_id:raise ValueError(f'Unknown provenance parent: {parent}')
            if kind=='T_split' and f['family']!='T':continue
            edges.append({'edge_type':kind,'parent_id':parent,'child_id':f['id']})
    return registry,edges,warnings


# Format only the allowed edge types as an indented forest, checking multiple
# parents/cycles. Source and inheritance forests are deliberately separate calls.
def _forest(registry, edges, allowed, families=None):
    fields=[f for f in registry if families is None or f['family'] in families]
    by_id={f['id']:f for f in fields};children={k:[] for k in by_id};parent={}
    for e in edges:
        if e['edge_type'] not in allowed:continue
        a,b=e['parent_id'],e['child_id']
        if a not in by_id or b not in by_id:raise ValueError('Wrong-family genealogy edge')
        if b in parent and parent[b]!=a:raise ValueError('Genealogy forest has multiple parents')
        parent[b]=a;children[a].append(b)
    # Sort siblings deterministically by birth time, lineage path and stable ID; labels
    # never define identity.
    def key(i):return (by_id[i]['birth_time'], by_id[i].get('lineage_path') or [],i)
    for cs in children.values():cs.sort(key=key)
    lines=[];visited=set()
    # Depth-first traversal appends readable labels and inactive markers; stack/visited
    # sets detect invalid cycles and duplicate visits.
    def visit(i,indent,stack):
        if i in stack:raise ValueError('Cycle in genealogy')
        if i in visited:return
        visited.add(i);f=by_id[i]
        lines.append('  '*indent+f"{f['display_label']} [{i}]"+(' (inactive)' if not f.get('active',True) else ''))
        for child in children[i]:visit(child,indent+1,stack|{i})
    for root in sorted((i for i in by_id if i not in parent),key=key):visit(root,0,set())
    if len(visited)!=len(by_id):raise ValueError('Cycle in genealogy')
    return '\n'.join(lines) or '(empty)'


# Build two forests and table rows, optionally export CSV/DOT, and return
# text/rows/edges/warnings. Reads metadata only, never reruns dynamics.
def analyze_genealogy(manifest, csv_path=None, graphviz=None):
    registry,edges,warnings=load_genealogy(manifest);by_id={f['id']:f for f in registry}
    source=_forest(registry,edges,{'T_split','P_source_birth'})
    inheritance=_forest(registry,edges,{'P_parameter_inheritance'},{'P'})
    rows=[]
    for f in registry:
        src=by_id.get(f.get('source_T_id'),{});par=by_id.get(f.get('inheritance_parent_id'),{})
        rows.append({'field_id':f['id'],'display_label':f['display_label'],'family':f['family'],
                     'lineage_path':f.get('lineage_path'),'source_T_id':src.get('id'),
                     'source_T_label':src.get('display_label'),'inheritance_parent_id':par.get('id'),
                     'inheritance_parent_label':par.get('display_label'),
                     **{k:f.get(k) for k in ('birth_time','death_time','birth_segment','generation','birth_event_id','source_birth_ordinal')}})
    if csv_path:
        p=Path(csv_path);p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('w',encoding='utf-8',newline='') as out:
            writer=csv.DictWriter(out,fieldnames=COLUMNS);writer.writeheader()
            for row in rows:writer.writerow({**row,'lineage_path':json.dumps(row['lineage_path'])})
    if graphviz:
        styles={'T_split':'solid','P_source_birth':'dashed','P_parameter_inheritance':'dotted'}
        lines=['digraph genealogy {','  rankdir=LR;']
        for f in registry:
            lines.append(f"  {json.dumps(f['id'])} [label={json.dumps(f['display_label']+' ['+f['id']+']')}];")
        for e in edges:
            lines.append(f"  {json.dumps(e['parent_id'])} -> {json.dumps(e['child_id'])} [style={styles[e['edge_type']]}, label={json.dumps(e['edge_type'])}];")
        p=Path(graphviz);p.parent.mkdir(parents=True,exist_ok=True);p.write_text('\n'.join(lines+['}'])+'\n',encoding='utf-8')
    text='SOURCE PROVENANCE (T split + T -> P; indentation means source-child)\n'+source+'\n\nPARAMETER INHERITANCE (P -> P only)\n'+inheritance
    if warnings:text+='\n\nNOTES\n'+'\n'.join(warnings)
    return {'text':text,'rows':rows,'edges':edges,'warnings':warnings}
