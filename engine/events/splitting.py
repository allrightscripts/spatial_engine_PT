"""Periodic-aware connected components and persistent isolation detection."""
from copy import deepcopy
import numpy as np
from scipy import ndimage
from .base import SplitEvent, MutationEvent
from .provenance import enabled, next_field_id, set_split


def components(u,threshold,periodic_axes):
    """Face connectivity (2 neighbours in 1D, 4 in 2D, 6 in 3D), including torus seams."""
    labels,n=ndimage.label(u>threshold)
    parent=np.arange(n+1)
    # Find and compress the union-find representative joining periodic boundary labels;
    # this is component identity, not genealogy.
    def root(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]]; a=parent[a]
        return a
    for axis in periodic_axes:
        a=np.take(labels,0,axis=axis).ravel(); b=np.take(labels,-1,axis=axis).ravel()
        for x,y in np.unique(np.stack((a,b),axis=1),axis=0):
            if x and y: parent[root(int(y))]=root(int(x))
    lookup=np.array([root(i) for i in range(n+1)])
    merged=lookup[labels]
    return [merged==i for i in np.unique(merged) if i]

class SplitDetector:
    # Validate cadence/threshold settings and initialize component-history caches; these
    # caches track isolation duration across checks.
    def __init__(self,config):
        self.config=config; self.tracks={}; self.last_counts={}; self.last_time={}
        if type(config.get('k_split_check',1)) is not int or config.get('k_split_check',1)<1 or config.get('tau_split',0)<0:
            raise ValueError('Invalid split cadence')
        if config.get('threshold',.1)<0 or config.get('min_cells',2)<1 or config.get('mutation_sigma',0)<0:
            raise ValueError('Invalid split threshold/size/mutation')
    # Find sufficiently large connected components in eligible active fields. Preserve
    # isolation age only for unambiguous component correspondence, then propose/apply a
    # split.
    def check(self,sim,t):
        c=self.config
        for i,f in enumerate(list(sim.registry)):
            if not f['active'] or f['family'] not in c.get('families',['T']): continue
            axes=[a for a,pair in enumerate(sim.grid.boundaries(f['bc'])) if pair[0]['type']=='periodic']
            masks=components(sim.u[i],c.get('threshold',.1),axes)
            masks=[m for m in masks if m.sum()>=c.get('min_cells',2)]
            self.last_counts[f['id']]=len(masks); self.last_time[f['id']]=t
            if len(masks)<2:
                self.tracks.pop(f['id'],None); continue
            old=self.tracks.get(f['id'],[])
            ages=[t]*len(masks)
            # Preserve duration only under a bijection of overlapping components.
            # A merge, new fragment, or ambiguous correspondence resets isolation.
            if len(old)==len(masks):
                overlap=np.array([[np.count_nonzero(m&o[0]) for o in old] for m in masks])
                positive=overlap>0
                if np.all(positive.sum(axis=0)==1) and np.all(positive.sum(axis=1)==1):
                    ages=[old[int(np.argmax(row))][1] for row in overlap]
            self.tracks[f['id']]=list(zip(masks,ages))
            if min(t-a for a in ages)+1e-12 >= c.get('tau_split',1):
                self.split(sim,i,masks,t)
                self.tracks.pop(f['id'],None)

    def split(self,sim,index,masks,t):
        """Partition without destroying mass; residual tails belong to largest lobe.

        Every cell keeps its location and value. Parent retires, daughters append
        to the global registry. This prevents any historic DOF renumbering.
        """
        f=sim.registry[index]; original=sim.u[index].copy()
        masks=[m.copy() for m in masks]
        ordering='legacy_component_order'
        if enabled(sim):
            # Order components by descending mass, Cartesian centroid and first flat cell
            # index. Evaluate before residual tails are assigned to the largest component.
            def key(mask):
                weights=np.maximum(original[mask],0);mass=float(np.sum(original[mask])*sim.grid.volume)
                centroid=tuple(float(np.average(x[mask],weights=weights)) if weights.sum()>0 else float(x[mask].mean()) for x in sim.grid.mesh)
                return (-mass,centroid,int(np.flatnonzero(mask)[0]))
            masks.sort(key=key)
            ordering='descending_component_mass_then_centroid_lexicographic_then_first_flat_index'
        largest=int(np.argmax([np.sum(original[m]) for m in masks]))
        masks[largest] |= ~np.logical_or.reduce(masks)
        children=[]; sigma=self.config.get('mutation_sigma',0)
        for branch,mask in enumerate(masks):
            label=(f['genealogy_label'].replace('_root','')+f'_{branch}')
            name=next_field_id(sim) if enabled(sim) else label
            if any(r['id']==name for r in sim.registry): name=sim.next_name(name)
            p=deepcopy(f['parameters'])
            sim.add_field(name,f['family'],original*mask,t,p,parent=f['id'],bc=f['bc'],label=label)
            set_split(sim.registry[-1],f,branch)
            children.append(name)
            if sigma:
                before=deepcopy(sim.rng.bit_generator.state)
                updated={k:float(v*np.exp(sim.rng.normal(0,sigma))) for k,v in p.items()}
                sim.apply(MutationEvent(t,source='split',targets=[name],parameters={'updates':updated},rng={'state_before':before,'sigma':sigma}))
        sim.retire(index,t)
        sim.genealogy.append({'parent':f['id'],'children':children,'time':t})
        sim.record(SplitEvent(t,source='split_detector',targets=children,parameters={'parent_id':f['id'],'child_ordering_rule':ordering,
            'parent_display_label':f['display_label'],'child_display_labels':[sim.registry[sim.index(n)]['display_label'] for n in children],'mass_before':float(original.sum()*sim.grid.volume),'mass_after':float(sum(sim.u[sim.index(n)].sum() for n in children)*sim.grid.volume)}))
