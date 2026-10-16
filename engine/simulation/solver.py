"""Explicit Heun integration with variable fields, exact scheduled event times.

No model family labels enter the deterministic integrator. Registry rows are
append-only; retired rows remain zero. Only active fields are sent to Model.rhs.
No clipping hides unstable steps: materially negative or nonfinite values fail.
"""
from copy import deepcopy
import numpy as np
from engine.archive import json_array, write_archive, pack_events
from .grid import Grid
from ..model.definitions import Model
from ..events.base import from_config, ExtinctionEvent
from ..events.operators import HANDLERS, spatial_mask, PoissonNucleation
from ..events.splitting import SplitDetector
from ..analysis.diagnostics import observe
from ..events.provenance import initialize, record_creation
from ..events.targeting import resolve_event

class Simulation:
    # Validate grid/time/seed, create the run-owned RNG, registry and initial fields, and
    # prepare scheduled events. Initial data is copied into physical arrays.
    def __init__(self,config,model=None,handlers=None,stochastic=None):
        self.config=deepcopy(config); c=self.config
        # JSON serialization also rejects nonfinite config values.
        json_array(c)
        self.grid=Grid(c['grid']['shape'],c['grid']['lengths'])
        if c.get('dimension',len(self.grid.shape)) != len(self.grid.shape): raise ValueError('dimension/grid mismatch')
        self.dt=float(c['dt']); self.end=float(c['t_end']); self.save_dt=float(c.get('save_dt',self.dt))
        if min(self.dt,self.end,self.save_dt)<=0: raise ValueError('Times must be positive')
        if 'seed' not in c or type(c['seed']) is not int or c['seed']<0: raise ValueError('Explicit nonnegative integer seed required')
        self.rng=np.random.default_rng(c['seed']); self.rng_initial=deepcopy(self.rng.bit_generator.state)
        self.model=model or (self._load_model(c) if 'model' in c else Model()); self.handlers={**HANDLERS,**(handlers or {})}
        self.stochastic=stochastic or PoissonNucleation(c.get('nucleation',{}))
        self.default_bc=c.get('boundary',{'type':'neumann','value':0})
        self.registry=[]; self.u=np.empty((0,)+self.grid.shape); self.events=[]; self.genealogy=[]; self.held=[]
        self.detector=SplitDetector(c.get('split',{})); self.times=[]; self.snapshots=[]; self.diagnostics=[]
        self.schedule=sorted([from_config(e) for e in c.get('events',[])],key=lambda e:e.time)
        for e in self.schedule:
            if e.time<0 or e.time>self.end: raise ValueError('Scheduled event outside run')
            if e.kind not in self.handlers: raise ValueError(f'No scheduled handler for {e.kind}; split is detector-driven')
        for f in c['initial_fields']:
            initial=f.get('initial',{'value':1})
            values=initial.get('value',1)*spatial_mask(self.grid,initial.get('geometry',{}))
            self.add_field(f['id'],f['family'],values,0,f.get('parameters',{}),bc=f.get('bc'))
            self.registry[-1].update(model_type=f.get('model_type','default'), birth_segment=0)
            if 'mobility_row' in f: self.registry[-1]['mobility_row']=deepcopy(f['mobility_row'])
        if not self.registry: raise ValueError('Need at least one initial field')
        self.cursor=0; self.used=False

    # Delegate configured model creation to the trusted model loader; kept separate so
    # initialization can also accept a supplied Model instance.
    @staticmethod
    def _load_model(config):
        from ..model.loading import load_model
        return load_model(config)

    # Require finite scalar field parameters, nonnegative diffusion and positive capacity.
    # This checks metadata, not numerical stability of a chosen PDE.
    @staticmethod
    def check_parameters(p):
        if any(not isinstance(v,(int,float)) or not np.isfinite(v) for v in p.values()): raise ValueError('Field parameters must be finite scalars')
        if p.get('diffusion',0)<0 or p.get('capacity',1)<=0: raise ValueError('Invalid diffusion/capacity')

    # Append a unique-ID registry row and physical field after validating
    # parameters/BC/data. Never renumber retired rows or existing field IDs.
    def add_field(self,name,family,values,t,parameters,parent=None,bc=None,label=None):
        if not isinstance(name,str) or not name or any(f['id']==name for f in self.registry): raise ValueError('Unique nonempty field ID required')
        self.check_parameters(parameters)
        boundary=deepcopy(bc if bc is not None else self.default_bc); self.grid.boundaries(boundary)
        values=np.broadcast_to(np.asarray(values,float),self.grid.shape).copy()
        if not np.isfinite(values).all() or (np.any(values<0) and not self.config.get('allow_signed_fields',False)):
            raise ValueError('Invalid initial field')
        generation=0 if parent is None else self.registry[self.index(parent)]['generation']+1
        self.registry.append({'id':name,'family':family,'parent_id':parent,'birth_time':float(t),'death_time':None,
            'generation':generation,'parameters':deepcopy(parameters),'initial_parameters':deepcopy(parameters),
            'active':True,'genealogy_label':label or name,'bc':boundary,
            'model_type':'default','birth_segment':getattr(self,'birth_segment',0)})
        initialize(self,self.registry[-1],parent)
        self.u=np.concatenate((self.u,values[None]),axis=0)

    # Resolve a stable field ID to its global registry row, including retired history;
    # unknown IDs raise ValueError.
    def index(self,name):
        for i,f in enumerate(self.registry):
            if f['id']==name: return i
        raise ValueError(f'Unknown target {name}')
    # Find an unused legacy prefix_integer name while reserving all historical IDs,
    # including inactive fields.
    def next_name(self,prefix):
        i=0
        while any(f['id']==f'{prefix}_{i}' for f in self.registry): i+=1
        return f'{prefix}_{i}'
    # Resolve explicit IDs or the legacy wildcard ['*'], returning only currently active
    # global registry indices.
    def target_indices(self,targets):
        ids=range(len(self.registry)) if targets==['*'] else [self.index(n) for n in targets]
        return [i for i in ids if self.registry[i]['active']]
    # Mark a field inactive with a death time and zero its current physical values,
    # retaining historical identity in the registry.
    def retire(self,i,t):
        self.registry[i]['active']=False; self.registry[i]['death_time']=float(t); self.u[i]=0
    # Assign a deterministic event ID, attach it to newly created fields and append a
    # serialized event record.
    def record(self,event):
        event.event_id=f'event_{len(self.events):08d}'
        record_creation(self,event)
        self.events.append(event.record())
    # Resolve semantic targets, invoke the existing event handler, then record its
    # resulting metadata. Handlers may mutate state and registry.
    def apply(self,event):
        resolve_event(self,event)
        self.handlers[event.kind](self,event); self.record(event)

    # Advance active grid fields by explicit Heun over h. Reject nonfinite/materially
    # negative values; only roundoff-scale negatives are clipped.
    def deterministic_step(self,t,h):
        ids=[i for i,f in enumerate(self.registry) if f['active']]
        if not ids: return
        fields=[self.registry[i] for i in ids]; u=self.u[ids]
        a=self.model.rhs(u,fields,self.grid,t)
        b=self.model.rhs(u+h*a,fields,self.grid,t+h)
        updated=u+h*(a+b)/2
        if updated.shape!=u.shape or not np.isfinite(updated).all(): raise FloatingPointError('Invalid deterministic state')
        if np.min(updated)<-1e-10: raise FloatingPointError('Negative density: reduce dt or revise model; no silent clipping')
        self.u[ids]=np.maximum(updated,0) # roundoff only

    # Apply queued scheduled events up to time t in stored order and advance the schedule
    # cursor once per event.
    def scheduled(self,t):
        while self.cursor<len(self.schedule) and self.schedule[self.cursor].time<=t+1e-12:
            self.apply(self.schedule[self.cursor]); self.cursor+=1

    # Save a copy of the current physical fields and cheap diagnostics at t; later births
    # do not resize earlier snapshots immediately.
    def snapshot(self,t):
        self.times.append(float(t)); self.snapshots.append(self.u.copy()); self.diagnostics.append(observe(self,t))

    # Execute the single-use FV loop, respecting event/save boundaries and held
    # interventions; return its standard archive dictionary.
    def run(self):
        if self.used: raise RuntimeError('Simulation instances are single-use')
        self.used=True; t=0.; step=0; save_index=1
        self.scheduled(t); self.detector.check(self,t); self.snapshot(t)
        while t<self.end-1e-13:
            next_event=self.schedule[self.cursor].time if self.cursor<len(self.schedule) else self.end
            stops=[t+self.dt,self.end,save_index*self.save_dt,next_event]
            stops += [a['end'] for a in self.held if a['end']>t+1e-12]
            target=min(stops); h=target-t
            if h<=0: raise RuntimeError('Nonadvancing timestep')
            self.deterministic_step(t,h); t=target; step+=1
            # Held interventions also reach descendants born while active.
            for a in self.held:
                roots=set(a['event'].targets)
                ids=[]
                for i,f in enumerate(self.registry):
                    if a['event'].parameters.get('target_selector') is not None:
                        if f['active'] and f['id'] in roots: ids.append(i)
                        continue
                    ancestor=f
                    while ancestor['parent_id'] is not None and ancestor['id'] not in roots:
                        ancestor=self.registry[self.index(ancestor['parent_id'])]
                    if f['active'] and ('*' in roots or ancestor['id'] in roots): ids.append(i)
                self.u[ids] *= 1-a['event'].strength*a['mask']
            remaining=[]
            for a in self.held:
                if a['end']<=t+1e-12:
                    e=deepcopy(a['event']); e.time=t; e.parameters={**e.parameters,'phase':'end','start_event_id':e.event_id}; self.record(e)
                else: remaining.append(a)
            self.held=remaining
            self.stochastic.step(self,h,t)
            self.scheduled(t)
            if step%self.config.get('split',{}).get('k_split_check',5)==0: self.detector.check(self,t)
            cutoff=self.config.get('extinction_mass',0)
            if cutoff>0:
                mass=self.u.reshape(len(self.registry),-1).sum(axis=1)*self.grid.volume
                for i,f in enumerate(self.registry):
                    if f['active'] and mass[i]<cutoff:
                        self.apply(ExtinctionEvent(t,source='mass_threshold',targets=[f['id']],parameters={'mass':float(mass[i]),'threshold':cutoff}))
            if not np.isfinite(self.u).all() or np.any(self.u < -1e-10): raise FloatingPointError('Invalid post-event state')
            if t>=save_index*self.save_dt-1e-12 or t>=self.end-1e-12:
                self.snapshot(t); save_index+=1
        self.steps=step
        return self.archive()

    def archive(self):
        """Pad history ONCE at serialization; no repeated expansion of snapshots."""
        n=len(self.registry); cells=int(np.prod(self.grid.shape)); nt=len(self.times)
        coeff=np.zeros((nt,n*cells)); mass=np.zeros((nt,n)); active=np.zeros((nt,n),np.int8)
        counts=np.full((nt,n),-1,dtype=np.int64); checked=np.full((nt,n),-1.)
        for k,(u,obs) in enumerate(zip(self.snapshots,self.diagnostics)):
            coeff[k,:u.size]=u.ravel(); size=len(u)
            mass[k,:size]=obs['mass']; active[k,:size]=obs['active']; counts[k,:size]=obs['components']; checked[k,:size]=obs['components_checked_at']
        fields=np.repeat([f['id'] for f in self.registry],cells)
        modes=np.tile(np.array(list(np.ndindex(self.grid.shape)),np.int64),(n,1))
        metadata={'representation':'cell_average','dimension':len(self.grid.shape),'method':'finite_volume_Heun',
            'steps':self.steps,'config':self.config,'snapshot_semantics':'post-event; first snapshot after t=0 schedule',
            'component_counts':'cached at components_checked_at; -1 means not checked',
            'nonnegative_density_model':True}
        data={'time':np.array(self.times),'coefficients':coeff,'initial_coefficients':coeff[0].copy(),
            'fields':fields,'modes':modes,'order_json':json_array(list(zip(fields.tolist(),modes.tolist()))),
            'parameters_json':json_array({f'Lx{i+1}':L for i,L in enumerate(self.grid.lengths)}),
            'metadata_json':json_array(metadata),'equations_text':np.array('dU_i/dt = div(sum_j L_ij(U) grad(mu_j)) + R_i(U); mu=delta F/delta U; plus logged events'),
            'grid_json':json_array({'shape':self.grid.shape,'lengths':self.grid.lengths,'placement':'cell_center','flatten_order':'C','basis':'piecewise_constant_cell_indicator'}),
            'field_metadata_json':json_array(self.registry),'genealogy_json':json_array(self.genealogy),
            'boundary_conditions_json':json_array({f['id']:f['bc'] for f in self.registry}),
            'model_metadata_json':json_array({'model':type(self.model).__name__,'functional':type(self.model.functional).__name__,'mobility':type(self.model.mobility).__name__,'reaction':type(self.model.reaction).__name__}),
            'rng_json':json_array({'seed':self.config['seed'],'initial_state':self.rng_initial,'final_state':self.rng.bit_generator.state,'bit_generator':type(self.rng.bit_generator).__name__,'numpy_version':np.__version__}),
            'observables_mass':mass,'observables_active':active,'observables_components':counts,'observables_components_checked_at':checked}
        for key in ('total_mass','active_P','active_T','entropy','functional','functional_available'):
            data['observables_'+key]=np.array([o[key] for o in self.diagnostics])
        data.update(pack_events(self.events,self.config.get('event_json_limit',1000)))
        for axis,x in enumerate(self.grid.coords): data[f'grid_x{axis+1}']=x
        return data
