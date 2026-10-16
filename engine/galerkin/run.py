"""Compose fixed-space Galerkin flows and explicit event maps into a full run.

The existing v0.1 registry, event operators and SplitDetector are reused. A
planning adapter queues events so the old archive is closed BEFORE mutation.
Adaptive solver trials never mutate registry, detectors or RNG state.
"""
from copy import deepcopy
import logging
from .. import __version__, MANIFEST_SCHEMA_VERSION
from engine.archive import SCHEMA_VERSION
import importlib
import json
from pathlib import Path
import shutil
import numpy as np
from engine.numerics.ode import integrate_provider
from engine.archive import write_archive, json_array, decode
from ..simulation.solver import Simulation
from ..events.base import PBirthEvent, LandscapeEvent, from_config, CatastropheEvent, ExtinctionEvent, SplitEvent
from ..events.splitting import SplitDetector
from ..events.operators import spatial_mask
from ..events.p_birth import p_birth
from ..events.hazard import BirthClock
from ..events.landscape import landscape
from .basis import TensorBasis
from .integration import IntegrationSession
from .provider import GalerkinProvider
from ..events.provenance import edges


# Write finite JSON via a temporary sibling file and atomic replacement. Parent
# directories are created; NaN/Inf are rejected.
def dump_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def normalize_event(raw,dimension):
    """Accept explicit v0.2 catastrophe controls and legacy geometry/parameters."""
    data=deepcopy(raw);p=data.setdefault('parameters',{})
    for key in ('duration','effect_type','close_segment','target_families'):
        if key in data:p[key]=data.pop(key)
    if 'shape' in data:
        geometry={'type':data.pop('shape')}
        for key in ('center','radius','radii','low','high'):
            if key in data:geometry[key]=data.pop(key)
        if 'axes' in data:geometry['radii']=data.pop('axes')
        if 'width' in data:
            width=np.broadcast_to(np.asarray(data.pop('width'),float),(dimension,))
            center=np.asarray(geometry['center'])
            geometry.update(low=(center-width/2).tolist(),high=(center+width/2).tolist())
        data['geometry']=geometry
    if 'mask' in data:
        data['geometry']={'mask':data.pop('mask'),'parameters':data.pop('mask_parameters',{})}
    if data.get('type')=='p_birth':
        for key in list(data):
            if key not in ('type','time','source','targets','geometry','strength','parameters','rng','event_id'):
                p[key]=data.pop(key)
        if 'policy' in p: p['parameter_policy']=p['policy']
    p['closure_policy']='all_state_or_rhs_changes_close_segment'
    return from_config(data)


class PlanningDetector(SplitDetector):
    # Initialize component tracking and an empty proposal list; planning must not yet
    # change the live field dimension.
    def __init__(self,config):
        super().__init__(config);self.proposals=[]
    # Queue the detected component masks for the next boundary commit, avoiding duplicate
    # component detection.
    def split(self,sim,index,masks,t):
        # Cache detected masks rather than performing component analysis twice.
        self.proposals.append((index,masks,t))


class EventState(Simulation):
    """Reuse registry/event code, but do not use v0.1 deterministic integration."""
    # Reuse registry/event machinery without FV time stepping; configure accessibility,
    # rate events and the split planner.
    def __init__(self,config):
        c=deepcopy(config);c['events']=[];c['allow_signed_fields']=True
        super().__init__(c)
        self.planning=False;self.pending=[];self.rates=[]
        self.landscape_base=np.broadcast_to(np.asarray(c.get('landscape',{}).get('accessibility',1.0),float),self.grid.shape).copy()
        if not np.isfinite(self.landscape_base).all() or np.any((self.landscape_base<0)|(self.landscape_base>1)):
            raise ValueError('Initial accessibility must be in [0,1]')
        self.accessibility=self.landscape_base.copy();self.landscape_layers=[]
        self.handlers={**self.handlers,'catastrophe':self.controlled,'sabotage':self.controlled,
                       'p_birth':p_birth,'landscape':landscape}
        self.detector=PlanningDetector(c.get('split',{}))

    # During planning, queue events; otherwise expand compatible family targets and
    # delegate handler application/recording. Conflicting target specifications fail.
    def apply(self,event):
        if self.planning:
            self.pending.append(event);return
        families=event.parameters.get('target_families',[])
        if families and event.parameters.get('target_selector') is not None:
            raise ValueError('target_selector conflicts with target_families')
        if families:
            event.targets=list(dict.fromkeys(event.targets+[f['id'] for f in self.registry
                                             if f['active'] and f['family'] in families]))
        super().apply(event)

    # Apply spectral impulse suppression or install/remove a finite-duration sink. Unlike
    # held FV suppression, a rate sink enters the continuous RHS.
    @staticmethod
    def controlled(sim,event):
        p=event.parameters
        if p.get('phase')=='end':
            sim.rates=[r for r in sim.rates if r['event_id']!=p['start_event_id']]
            return
        duration=float(p.get('duration',0));effect=p.get('effect_type','rate' if duration>0 else 'impulse')
        if duration<0 or not np.isfinite(duration):raise ValueError('Invalid catastrophe duration')
        mask=spatial_mask(sim.grid,event.geometry,event.time)
        ids=sim.target_indices(event.targets)
        if effect=='impulse':
            if duration!=0:raise ValueError('impulse has duration=0; use effect_type=rate for a finite-duration sink')
            if not 0<=event.strength<=1:raise ValueError('Impulse suppression strength must be [0,1]')
            sim.u[ids]*=1-event.strength*mask
        elif effect=='rate':
            if duration<=0 or event.strength<0:raise ValueError('rate needs positive duration and nonnegative strength')
            # record() immediately assigns this same deterministic event_id.
            sim.rates.append({'event_id':f'event_{len(sim.events):08d}','end':event.time+duration,
                              'roots':[sim.registry[i]['id'] for i in ids],
                              'geometry':deepcopy(event.geometry),'mask_values':mask.tolist(),'strength':event.strength,
                              'resolved_only':p.get('target_selector') is not None})
        else:raise ValueError('Catastrophe effect_type must be impulse or rate')

    def active_forcing(self):
        """Resolve active descendants of the original target roots once per segment."""
        result=[]
        for r in self.rates:
            targets=[]
            for f in self.registry:
                if r.get('resolved_only'):
                    if f['active'] and f['id'] in r['roots']:targets.append(f['id'])
                    continue
                ancestor=f
                while ancestor['parent_id'] is not None and ancestor['id'] not in r['roots']:
                    ancestor=self.registry[self.index(ancestor['parent_id'])]
                if f['active'] and ancestor['id'] in r['roots']:targets.append(f['id'])
            result.append({'geometry':r['geometry'],'mask_values':r['mask_values'],'strength':r['strength'],'targets':targets})
        return result


class SegmentedRun:
    # Validate run settings, initialize event state/clock, and create the output manifest
    # and model-source snapshot. The destination must be empty.
    def __init__(self,config,destination):
        self.config=deepcopy(config);json_array(self.config)
        if config.get('solver',{}).get('backend','spectral_galerkin')!='spectral_galerkin':
            raise ValueError('SegmentedRun is the spectral backend; tools.run_finite_volume retains finite volume')
        self.path=Path(destination)
        if self.path.exists() and any(self.path.iterdir()):
            raise FileExistsError('Run directory must be empty; existing runs are never overwritten')
        self.path.mkdir(parents=True,exist_ok=True)
        self.state=EventState(config);self.settings=config['solver']
        self.birth_clock=BirthClock(config.get('stochastic_events',{}).get('p_birth',{}),self.state.rng,config['seed'])
        self.birth_due=False
        self.schedule=sorted([normalize_event(e,len(self.state.grid.shape)) for e in config.get('events',[])],key=lambda e:e.time)
        if any(e.time<0 or e.time>self.state.end for e in self.schedule):raise ValueError('Event outside run interval')
        if any(e.kind=='split' for e in self.schedule):raise ValueError('SplitEvent is generated by the detector')
        self.cursor=0;self.segment_number=0
        self.manifest={'schema_version':MANIFEST_SCHEMA_VERSION,
            'run_manifest_schema_version':MANIFEST_SCHEMA_VERSION,'software_version':__version__,
            'archive_schema_version':SCHEMA_VERSION,'run_id':config.get('run_id','RUN_001'),
            'status':'running','global_config':self.config,'seed':config['seed'],'segments':[],
            'rng':{'seed':config['seed'],'bit_generator':type(self.state.rng.bit_generator).__name__,'numpy_version':np.__version__},
            'field_registry':[],'genealogy':[],'events':[],'relations':[]}
        dump_json(self.path/'config.json',config)
        module=importlib.import_module('engine.model.'+config.get('model','cultural_v01'))
        (self.path/'model_source').mkdir()
        shutil.copy2(module.__file__,self.path/'model_source'/Path(module.__file__).name)
        for name in ('event_rules.py','masks.py','mobility.py','birth_profiles.py'):
            source=Path(__file__).resolve().parents[1]/'model'/name
            shutil.copy2(source,self.path/'model_source'/name)
        self.projection_in=None;self.incoming=[]

    # Refresh registry, typed genealogy edges, events and clock audit from live state,
    # then atomically save manifest.json.
    def refresh_manifest(self):
        self.manifest['genealogy_edges']=edges(self.state.registry)
        self.manifest['provenance_schema_version']='1.0'
        self.manifest.update(field_registry=deepcopy(self.state.registry),genealogy=deepcopy(self.state.genealogy),
                             events=deepcopy(self.state.events),rng_state=deepcopy(self.state.rng.bit_generator.state))
        if self.birth_clock.threshold is not None:
            self.manifest['birth_clock']={**self.birth_clock.audit(),'enabled':self.birth_clock.enabled}
        dump_json(self.path/'manifest.json',self.manifest)

    def build_segment(self,t):
        """Explicit physical→coefficient map, once after every event batch."""
        self.ids=[i for i,f in enumerate(self.state.registry) if f['active']]
        if not self.ids:return False
        fields=[self.state.registry[i] for i in self.ids]
        basis=TensorBasis.from_config(self.state.grid,fields,self.settings)
        self.provider=GalerkinProvider(self.config,fields,basis,self.state.active_forcing(),
            self.state.accessibility if self.state.landscape_layers or np.any(self.state.landscape_base!=1) else None)
        physical=self.state.u[self.ids].copy()
        self.a=basis.project(physical).ravel()
        reconstructed=basis.reconstruct(self.a.reshape((len(fields),)+basis.shape))
        diff=physical-reconstructed
        self.projection_in={'operation':'reconstruct -> event map -> project',
            'l2_error':float(np.sqrt(np.sum(diff*diff)*self.state.grid.volume)),
            'linf_error':float(np.max(np.abs(diff))),
            'mass_before':(physical.reshape(len(fields),-1).sum(axis=1)*self.state.grid.volume).tolist(),
            'mass_after':(reconstructed.reshape(len(fields),-1).sum(axis=1)*self.state.grid.volume).tolist(),
            'minimum_reconstructed':float(reconstructed.min()),'clipped':False}
        self.state.u[self.ids]=reconstructed
        self.start=t;self.parts=[];self.sample_diagnostics=[]
        self.initial_a=self.a.copy()
        self.integration_session=(IntegrationSession() if
            self.settings.get('reuse_solver',True) else None)
        return True

    def advance(self,t,target):
        """Integrate a control slab of the SAME ODE; no re-projection/re-solving."""
        # Detector cadence is independent of saved frames and adaptive trial steps.
        save_dt=self.config.get('save_dt',.05)
        points=np.arange(int(np.floor(t/save_dt))+1,int(np.ceil(target/save_dt)))*save_dt
        times=np.r_[t,points[(points>t+1e-12)&(points<target-1e-12)],target]
        data=integrate_provider(self.provider,self.a,t,target,
            method=self.settings.get('integrator','Radau'),rtol=self.settings.get('rtol',1e-7),
            atol=self.settings.get('atol',1e-9),max_step=self.settings.get('max_step',self.config['dt']),
            save_dt=save_dt,output_times=times,
            integration_session=self.integration_session,
            integral_clock=self.birth_clock.integral_clock(self.provider) if self.birth_clock.enabled else None)
        if self.birth_clock.enabled:
            clock_result=decode(data,'metadata_json')['integral_clock']
            self.birth_clock.value=clock_result['value'];self.birth_due=clock_result['reached']
        if float(data['time'][-1])<=t:
            raise RuntimeError('Event waiting time is below floating-point time resolution; reduce intensity')
        offset=0 if not self.parts else 1
        for when,a in zip(data['time'][offset:],data['coefficients'][offset:]):
            U=self.provider.basis.reconstruct(a.reshape((len(self.ids),)+self.provider.basis.shape))
            mass=U.reshape(len(self.ids),-1).sum(axis=1)*self.state.grid.volume
            energy=self.provider.model.functional.value(U,self.provider.fields,self.state.grid,float(when))
            positive=np.maximum(mass,0);total=positive.sum()
            fractions=positive[positive>0]/total if total>0 else np.array([])
            self.sample_diagnostics.append({'mass':mass,'functional':0. if energy is None else float(energy),
                'functional_available':int(energy is not None),'entropy':float(-np.sum(fractions*np.log(fractions))),
                'active_P':sum(f['family']=='P' for f in self.provider.fields),
                'active_T':sum(f['family']=='T' for f in self.provider.fields),
                'components':[self.state.detector.last_counts.get(f['id'],-1) for f in self.provider.fields],
                'components_checked_at':[self.state.detector.last_time.get(f['id'],-1) for f in self.provider.fields]})
        self.parts.append(data);self.a=data['coefficients'][-1].copy()
        self.state.u[self.ids]=U
        return float(data['time'][-1])

    def close_segment(self,t,reasons,event_out):
        """Write one immutable mathematical system and all its accumulated samples."""
        if not self.parts:return
        data=dict(self.parts[0])
        data['time']=np.concatenate([p['time'] if i==0 else p['time'][1:] for i,p in enumerate(self.parts)])
        data['coefficients']=np.concatenate([p['coefficients'] if i==0 else p['coefficients'][1:] for i,p in enumerate(self.parts)])
        data['initial_coefficients']=self.initial_a.copy()
        meta=decode(data,'metadata_json')
        last_meta=decode(self.parts[-1],'metadata_json')
        if 'integral_clock' in last_meta: meta['integral_clock']=last_meta['integral_clock']
        for counter in ('nfev','njev','nlu'):meta[counter]=sum(decode(p,'metadata_json')[counter] for p in self.parts)
        meta.update(segment_id=self.segment_number,closure_reasons=reasons,
                    projection_in=self.projection_in,control_slabs=len(self.parts),
                    endpoint_semantics='closed pre-event endpoint; next segment starts post-event')
        data['metadata_json']=json_array(meta)
        # Cached while advancing; no trajectory reconstruction or RHS repetition here.
        for key in self.sample_diagnostics[0]:
            data['observables_'+key]=np.array([item[key] for item in self.sample_diagnostics])
        data['observables_total_mass']=data['observables_mass'].sum(axis=1)
        data['rng_json']=json_array({'seed':self.config['seed'],'state_at_segment_close':self.state.rng.bit_generator.state})
        name=f'segment_{self.segment_number:03d}';folder=self.path/name;folder.mkdir()
        (folder/'equations.txt').write_text(self.provider.source(),encoding='utf-8')
        write_archive(folder/'trajectory.npz',data)
        entry={'id':self.segment_number,'path':name,'archive':name+'/trajectory.npz',
               'equations':name+'/equations.txt','start_time':self.start,'end_time':t,
               'field_ids':[f['id'] for f in self.provider.fields],'dof':len(self.provider.order),
               'closure_reasons':reasons,'projection_in':self.projection_in}
        dump_json(folder/'segment_metadata.json',entry)
        dump_json(folder/'event_in.json',self.incoming);dump_json(folder/'event_out.json',event_out)
        logging.getLogger(__name__).info(
            'segment=%03d time=[%g,%g] fields=%d closure=%s archive=%s',
            self.segment_number, self.start, t, len(self.ids), ','.join(reasons), folder/'trajectory.npz')
        self.manifest['segments'].append(entry);self.segment_number+=1
        self.refresh_manifest()

    def boundary_proposals(self,t,check_split,dt):
        """Plan all mutations without altering the saved pre-event coefficients."""
        s=self.state;s.pending=[];s.planning=True
        for layer in s.landscape_layers:
            if layer['end'] is not None and layer['end']<=t+1e-12:
                s.pending.append(LandscapeEvent(t,source='duration_end',parameters={'phase':'end','start_event_id':layer['event_id']}))
        for r in s.rates:
            if r['end']<=t+1e-12:
                s.pending.append(CatastropheEvent(t,source='duration_end',parameters={'phase':'end','start_event_id':r['event_id']}))
        while self.cursor<len(self.schedule) and self.schedule[self.cursor].time<=t+1e-12:
            s.pending.append(self.schedule[self.cursor]);self.cursor+=1
        if self.birth_due:
            s.pending.append(PBirthEvent(t,source='integrated_hazard',rng=self.birth_clock.audit()))
        if check_split:
            s.stochastic.step(s,dt,t)
            c=self.config.get('stochastic_catastrophe',{});rate=float(c.get('rate',0))
            if rate<0:raise ValueError('Negative catastrophe intensity')
            before=deepcopy(s.rng.bit_generator.state)
            for draw in range(int(s.rng.poisson(rate*dt)) if rate else 0):
                raw=deepcopy(c.get('event',{}));raw.update(time=t,type='catastrophe',source='poisson_catastrophe')
                raw.setdefault('shape','gaussian');raw['center']=[float(s.rng.uniform(0,L)) for L in s.grid.lengths]
                raw['rng']={'state_before':before,'dt':dt,'draw':draw}
                s.pending.append(normalize_event(raw,len(s.grid.shape)))
            # Do not use pre-jump component masks to split a post-jump field.
            if any(e.kind in ('catastrophe','sabotage','manual') and e.parameters.get('phase')!='end' for e in s.pending):
                s.detector.tracks.clear()
            elif self.config.get('split',{}).get('enabled',True):
                s.detector.check(s,t)
            cutoff=self.config.get('extinction_mass',0)
            if cutoff>0:
                masses=s.u.reshape(len(s.registry),-1).sum(axis=1)*s.grid.volume
                for i,f in enumerate(s.registry):
                    if f['active'] and masses[i]<cutoff:s.pending.append(ExtinctionEvent(t,source='mass_threshold',targets=[f['id']]))
        s.planning=False
        return s.pending,s.detector.proposals

    # Structural changes happen only at boundaries. Do not append fields inside
    # a provider RHS: its state dimension and coefficient order must stay fixed
    # until the current segment has been closed and archived.
    def commit(self,t,events,splits):
        """Event map acts in physical space; next build_segment applies projection."""
        s=self.state;first=len(s.events)
        s.birth_segment=self.segment_number
        for e in events:
            s.apply(e)
            if e.kind=='p_birth' and e.source=='integrated_hazard':
                self.birth_due=False;self.birth_clock.consumed()
        for index,masks,when in splits:
            if s.registry[index]['active']:
                SplitDetector.split(s.detector,s,index,masks,when)
        s.detector.proposals=[]
        self.incoming=deepcopy(s.events[first:])
        if not np.isfinite(s.u).all():raise FloatingPointError('Nonfinite event result')
        return self.incoming

    # Advance to control/event boundaries, close changed segments and commit event maps.
    # Return the final manifest; record failed status before re-raising errors.
    def run(self):
        s=self.state;t=0.;tick=0
        interval=float(self.config.get('split',{}).get('check_dt',s.dt*self.config.get('split',{}).get('k_split_check',5)))
        if interval<=0 or self.config.get('save_dt',.05)<=0:raise ValueError('Positive cadence required')
        try:
            events,splits=self.boundary_proposals(0,False,0)
            self.commit(0,events,splits)
            if not self.build_segment(0):raise ValueError('No active initial fields')
            while t<s.end-1e-12:
                next_tick=(tick+1)*interval
                next_event=self.schedule[self.cursor].time if self.cursor<len(self.schedule) else s.end
                ends=[r['end'] for r in s.rates if r['end']>t+1e-12]
                ends += [r['end'] for r in s.landscape_layers if r['end'] is not None and r['end']>t+1e-12]
                target=min([next_tick,next_event,s.end]+ends)
                t=self.advance(t,target)
                check=abs(t-next_tick)<1e-10
                if check:tick+=1
                events,splits=self.boundary_proposals(t,check,interval)
                changes=bool(events or splits)
                reasons=[e.kind+('_end' if e.parameters.get('phase')=='end' else '') for e in events]+(['split'] if splits else [])
                if changes or t>=s.end-1e-12:
                    planned=[e.record() for e in events]+[{'event_type':'split','time':t,'parent_id':s.registry[i]['id']} for i,_,_ in splits]
                    self.close_segment(t,reasons or ['end_time'],planned)
                    if changes:
                        old=self.segment_number-1;committed=self.commit(t,events,splits)
                        dump_json(self.path/f'segment_{old:03d}'/'event_out.json',committed)
                        relation={'from':old,'to':None,'time':t,'events':[e['event_id'] for e in committed],
                                  'map':'reconstruct -> event -> project'}
                        if t<s.end-1e-12 and self.build_segment(t):
                            relation['to']=self.segment_number;relation['projection']=self.projection_in
                        else:
                            # Preserve terminal post-event physical state, without inventing
                            # a zero-length continuous-flow segment or zero-field fake ODE.
                            np.savez_compressed(self.path/'terminal_state.npz',time=np.array(t),values=s.u,
                                                fields=np.array([f['id'] for f in s.registry]),
                                                active=np.array([f['active'] for f in s.registry],dtype=np.int8),
                                                grid_shape=np.array(s.grid.shape),lengths=np.array(s.grid.lengths))
                            self.manifest['terminal_state']='terminal_state.npz'
                            self.manifest['end_time_actual']=t
                            self.manifest['termination_reason']='end_time' if t>=s.end-1e-12 else 'no_active_fields'
                            self.manifest['relations'].append(relation);break
                        self.manifest['relations'].append(relation)
                    self.refresh_manifest()
            self.manifest['status']='completed';self.refresh_manifest()
            return self.manifest
        except Exception as exc:
            self.manifest['status']='failed';self.manifest['error']=str(exc);self.refresh_manifest();raise
