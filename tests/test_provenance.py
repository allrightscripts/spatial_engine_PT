from tests.support import config_path
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from engine.galerkin.run import EventState,SegmentedRun,normalize_event
from engine.events.splitting import SplitDetector
from engine.events.provenance import edges
from engine.events.targeting import resolve_selector
from engine.events.birth_source import select_source
from engine.model.event_rules import source_hazards,evaluate_birth_rate,SOURCE_RATES
from engine.events.hazard import BirthClock
from engine.analysis.genealogy import analyze_genealogy
from engine.archive import read_archive,decode

ROOT=Path(__file__).resolve().parents[1]


def config():
    c=json.loads((config_path('spectral_single.json')).read_text())
    c.update(t_end=.15,save_dt=.05,seed=2718)
    c['grid']={'shape':[24],'lengths':[2.]};c['solver'].update(modes=[8],integrator='DOP853',max_step=.01)
    c['initial_fields'][0]['parameters']={'diffusion':.001}
    c['split'].update(enabled=False,mutation_sigma=0)
    c['stochastic_events']={'p_birth':{'enabled':False,'source_mode':'per_T','birth_profile':'inside_T',
        'birth_profile_parameters':{'amplitude':1.,'power':3.,'epsilon':1e-12},
        'rate_parameters':{'lambda0':1.,'alpha':0.},'parameter_policy':'manual','diffusion':.001}}
    return c


def create(s,source=None,**kwargs):
    raw={'type':'p_birth','time':.02,'diffusion':.001,**kwargs}
    if source is not None:raw['source_T_id']=source
    e=normalize_event(raw,len(s.grid.shape));s.apply(e)
    return s.registry[-1],e


def two_societies():
    s=EventState(config());x=s.grid.mesh[0]
    SplitDetector({}).split(s,0,[x<1,x>=1],.01)
    return s,s.registry[1],s.registry[2]


class Provenance09Tests(unittest.TestCase):
    def test_three_siblings_no_implied_inheritance(self):
        s=EventState(config());children=[]
        for k in range(3):
            child,e=create(s,'T_root');children.append(child)
            self.assertEqual(child['display_label'],f'P_root;{k}')
            self.assertEqual(child['lineage_path'],[k]);self.assertIsNone(child['inheritance_parent_id'])
            self.assertEqual(child['source_T_id'],'T_root')
            self.assertEqual(child['birth_event_id'],e.event_id)
            self.assertNotEqual(child['id'],child['display_label'])
            self.assertEqual(child['id'],f'field_{k+1:06d}')
        self.assertEqual(len([e for e in edges(s.registry) if e['edge_type']=='P_source_birth']),3)
        self.assertFalse(any(e['edge_type']=='P_parameter_inheritance' for e in edges(s.registry)))
        s.retire(s.index(children[0]['id']),.03)
        self.assertEqual(create(s,'T_root')[0]['source_birth_ordinal'],3)

    def test_profile_from_selected_source_only_and_landscape(self):
        s,a,b=two_societies();s.accessibility[1]=0
        old=s.u.copy();child,event=create(s,a['id'])
        T=np.maximum(old[s.index(a['id'])],0);expected=(T/(T.max()+1e-12))**3*s.accessibility
        np.testing.assert_allclose(s.u[-1],expected,atol=1e-14)
        np.testing.assert_array_equal(s.u[-1][old[s.index(b['id'])]>0],0)
        self.assertEqual(s.u[-1][1],0);np.testing.assert_array_equal(s.u[:-1],old)
        self.assertEqual(child['source_T_lineage'],[0])
        for key in ('child_id','child_display_label','source_T_id','source_T_display_label','source_T_lineage',
                    'inheritance_parent_id','inheritance_parent_display_label','inheritance_parent_lineage',
                    'source_hazard','total_hazard','source_selection_probability','birth_profile',
                    'birth_profile_parameters','birth_time','birth_segment','resulting_parameters','birth_event_id'):
            self.assertIn(key,event.parameters)

    def test_source_probabilities_three_fields_and_zero_mass(self):
        s=EventState(config());s.add_field('other','T',2,0,{'diffusion':.001});s.add_field('empty','T',0,0,{'diffusion':.001})
        c=s.config['stochastic_events']['p_birth'];indices,rates=source_hazards(c,s.u,s.registry,s.grid,0)
        np.testing.assert_allclose(rates,[2,4,0]);p=rates/rates.sum()
        self.assertAlmostEqual(p.sum(),1);self.assertEqual(p[-1],0)
        self.assertEqual(evaluate_birth_rate(c,s.u,s.registry,s.grid,0),6)
        before=deepcopy(s.rng.bit_generator.state);expected=np.random.default_rng();expected.bit_generator.state=before
        chosen=int(expected.choice(3,p=p))
        e=normalize_event({'type':'p_birth','time':0},1);index,audit=select_source(s,c,e)
        self.assertEqual(index,indices[chosen]);self.assertEqual(audit['source_selection_probability'],p[chosen])
        self.assertEqual(s.rng.bit_generator.state,expected.bit_generator.state)

    def test_no_eligible_source_fails_without_creation(self):
        s=EventState(config());s.u[:]=0
        self.assertEqual(evaluate_birth_rate(s.config['stochastic_events']['p_birth'],s.u,s.registry,s.grid,0),0)
        before=deepcopy(s.rng.bit_generator.state)
        with self.assertRaisesRegex(ValueError,'no positive source'):create(s)
        self.assertEqual(len(s.registry),1);self.assertEqual(s.rng.bit_generator.state,before)

    def test_inheritance_crosses_source_branch(self):
        s,a,b=two_societies();p0,_=create(s,a['id'],field_parameters={'foo':7})
        p1,e=create(s,b['id'],parameter_policy='inherit_mutate',mobility_policy='inherit_mutate',parent_id=p0['id'],mutation_sigma=0,mobility_mutation_sigma=0)
        self.assertEqual(p1['source_T_id'],b['id']);self.assertEqual(p1['inheritance_parent_id'],p0['id'])
        self.assertEqual(p1['display_label'],'P_1;0');self.assertEqual(p1['parameters']['foo'],7)
        self.assertEqual(e.parameters['source_T_display_label'],'T_1')
        kinds={x['edge_type'] for x in edges(s.registry) if x['child_id']==p1['id']}
        self.assertEqual(kinds,{'P_source_birth','P_parameter_inheritance'})

    def test_first_birth_inheritance_fallback_keeps_source(self):
        s=EventState(config());p,e=create(s,'T_root',parameter_policy='inherit_mutate',fallback='random',parameter_ranges={'growth':[.01,.03]})
        self.assertIsNone(p['inheritance_parent_id']);self.assertEqual(p['source_T_id'],'T_root')
        self.assertEqual(e.parameters['parameter_creation']['effective'],'random')

    def test_inheritance_policy_alias(self):
        results=[]
        for key in ('parent_policy','inheritance_parent_policy'):
            s=EventState(config());create(s,'T_root',birth_profile_parameters={'amplitude':1});create(s,'T_root',birth_profile_parameters={'amplitude':2})
            child,e=create(s,'T_root',parameter_policy='inherit_mutate',**{key:'weighted_by_mass'})
            results.append((child,e.rng))
        self.assertEqual(results[0],results[1])

    def test_split_three_children_deterministic_mass_centroid(self):
        outputs=[]
        for reverse in (False,True):
            s,a,b=two_societies();i=s.index(b['id']);x=s.grid.mesh[0]
            masks=[(x>=1)&(x<1.333),(x>=1.333)&(x<1.667),x>=1.667]
            s.u[i,masks[0]]=1;s.u[i,masks[1]]=3;s.u[i,masks[2]]=2
            before=s.u[i].copy();ids_before=[f['id'] for f in s.registry]
            SplitDetector({}).split(s,i,list(reversed(masks)) if reverse else masks,.03)
            children=s.registry[-3:]
            self.assertEqual([f['lineage_path'] for f in children],[[1,0],[1,1],[1,2]])
            self.assertEqual([f['display_label'] for f in children],['T_1.0','T_1.1','T_1.2'])
            np.testing.assert_allclose(s.u[-3:].sum(axis=0),before)
            self.assertEqual([f['id'] for f in s.registry[:3]],ids_before)
            self.assertIn('descending_component_mass',s.events[-1]['parameters']['child_ordering_rule'])
            self.assertTrue(all(f['birth_event_id']==s.events[-1]['event_id'] for f in children))
            outputs.append((deepcopy(children),s.u[-3:].copy()))
        self.assertEqual(outputs[0][0],outputs[1][0]);np.testing.assert_array_equal(outputs[0][1],outputs[1][1])

    def test_equal_mass_centroid_tie_break(self):
        s=EventState(config());x=s.grid.mesh[0]
        SplitDetector({}).split(s,0,[x>=1,x<1],.01)
        self.assertTrue(np.all(s.u[1,x>=1]==0));self.assertEqual(s.registry[1]['display_label'],'T_0')

    def test_historical_source_survives_split(self):
        s=EventState(config());p,_=create(s,'T_root');old=deepcopy(p);u=s.u[-1].copy();x=s.grid.mesh[0]
        SplitDetector({}).split(s,0,[x<1,x>=1],.03)
        self.assertEqual(s.registry[1],old);np.testing.assert_array_equal(s.u[1],u)
        self.assertEqual(p['source_T_lineage_at_birth'],[])

    def test_multiple_roots_are_unambiguous(self):
        s=EventState(config());s.add_field('second_root','T',1,0,{'diffusion':.001})
        p,_=create(s,'T_root');q,_=create(s,'second_root')
        self.assertNotEqual(p['display_label'],q['display_label']);self.assertNotEqual(p['lineage_root_id'],q['lineage_root_id'])

    def test_selectors_all_family_exact(self):
        for selector,affected in [({'mode':'all_active'},[0,1]),({'mode':'family','families':['T']},[0]),({'mode':'family','families':['T','P']},[0,1])]:
            s=EventState(config());create(s,'T_root');before=s.u.copy()
            e=normalize_event({'type':'catastrophe','time':.05,'strength':.5,'target_selector':selector},1);s.apply(e)
            expected=before.copy();expected[affected]*=.5;np.testing.assert_allclose(s.u,expected)
            self.assertEqual(e.parameters['resolved_target_ids'],[s.registry[i]['id'] for i in affected])
            self.assertEqual(e.parameters['requested_selector'],selector)
        s=EventState(config());create(s,'T_root');before=s.u[1].copy()
        s.apply(normalize_event({'type':'catastrophe','time':.05,'strength':1.,'targets':['T_root']},1))
        np.testing.assert_array_equal(s.u[1],before)

    def test_subtree_follows_source_not_inheritance(self):
        s,a,b=two_societies();p,_=create(s,a['id']);q,_=create(s,b['id'],parent_id=p['id'],parameter_policy='inherit_mutate')
        selector={'mode':'lineage_subtree','root_field_id':a['id'],'include_root':True,'include_descendants':True}
        self.assertEqual([f['id'] for f in resolve_selector(s.registry,selector)],[a['id'],p['id']])
        selector['include_root']=False
        self.assertEqual([f['id'] for f in resolve_selector(s.registry,selector)],[p['id']])
        selector.update(include_root=True,include_descendants=False)
        self.assertEqual([f['id'] for f in resolve_selector(s.registry,selector)],[a['id']])
        self.assertEqual(len(resolve_selector(s.registry,{'mode':'lineage_subtree','root_field_id':'T_root'})),4)

    def test_selector_conflict_and_unknown_mode(self):
        s=EventState(config())
        for raw in ({'mode':'bad'},{'mode':'family','families':'T'},{'mode':'lineage_subtree','root_field_id':'missing'}):
            with self.assertRaises(ValueError):resolve_selector(s.registry,raw)
        with self.assertRaises(ValueError):s.apply(normalize_event({'type':'catastrophe','time':0,'targets':['T_root'],'target_selector':{'mode':'all_active'}},1))

    def test_duration_selector_is_snapshot_not_inheritance(self):
        s=EventState(config());p,_=create(s,'T_root')
        s.apply(normalize_event({'type':'catastrophe','time':.04,'strength':.5,'duration':.1,'effect_type':'rate','target_selector':{'mode':'family','families':['P']}},1))
        q,_=create(s,'T_root',parent_id=p['id'],parameter_policy='inherit_mutate')
        self.assertEqual(s.active_forcing()[0]['targets'],[p['id']])

    def test_hazard_sum_timing_and_reproducibility(self):
        c=config();c['stochastic_events']['p_birth'].update(enabled=True,max_births=2,rate_parameters={'lambda0':50.,'alpha':0.})
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=SegmentedRun(c,root/'a').run();b=SegmentedRun(c,root/'b').run()
            self.assertEqual(a['events'],b['events']);self.assertEqual(a['field_registry'],b['field_registry'])
            self.assertEqual(len(a['segments']),3);self.assertEqual(a['archive_schema_version'],'1.0')
            self.assertAlmostEqual(a['events'][0]['time'],np.random.default_rng(c['seed']).exponential()/100.,places=10)
            self.assertEqual({e['edge_type'] for e in a['genealogy_edges']},{'P_source_birth'})
            for x,y in zip(a['segments'],b['segments']):
                ax=read_archive(root/'a'/x['archive']);by=read_archive(root/'b'/y['archive'])
                np.testing.assert_array_equal(ax['coefficients'],by['coefficients'])
                self.assertTrue(all('birth_event_id' in f for f in decode(ax,'field_metadata_json')))
            report=analyze_genealogy(root/'a/manifest.json',root/'genealogy.csv',root/'genealogy.dot')
            self.assertEqual(len(report['rows']),3);self.assertIn('P_root;0',report['text'])
            self.assertIn('style=dashed',(root/'genealogy.dot').read_text())
            with (root/'genealogy.csv').open() as stream:self.assertEqual(len(list(csv.DictReader(stream))),3)

    def test_genealogy_separate_forests_and_legacy_unknown_source(self):
        s,a,b=two_societies();p,_=create(s,a['id']);q,_=create(s,b['id'],parent_id=p['id'],parameter_policy='inherit_mutate')
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'manifest.json';path.write_text(json.dumps({'field_registry':s.registry,'provenance_schema_version':'1.0'}))
            r=analyze_genealogy(path,graphviz=Path(d)/'g.dot')
            self.assertEqual(len([e for e in r['edges'] if e['edge_type']=='P_parameter_inheritance']),1)
            self.assertIn('style=dotted',(Path(d)/'g.dot').read_text())
            legacy=[{'id':'T_root','family':'T','parent_id':None,'birth_time':0,'active':True},
                    {'id':'P_0','family':'P','parent_id':None,'birth_time':1,'active':True}]
            path.write_text(json.dumps({'field_registry':legacy}))
            r=analyze_genealogy(path);self.assertEqual(r['edges'],[]);self.assertTrue(r['warnings'])

    def test_unknown_rate_and_negative_rates_fail(self):
        s=EventState(config());c=s.config['stochastic_events']['p_birth']
        with self.assertRaises(ValueError):source_hazards({**c,'source_rate_function':'missing'},s.u,s.registry,s.grid,0)
        with self.assertRaises(ValueError):source_hazards({**c,'rate_parameters':{'lambda0':-1}},s.u,s.registry,s.grid,0)

    def test_per_T_profile_small_source_does_not_fallback_outside(self):
        s=EventState(config());s.u[0]=1e-14
        with self.assertRaisesRegex(ValueError,'refusing uniform'):create(s,'T_root')
        self.assertEqual(len(s.registry),1)

    def test_3d_per_T_segment_projection(self):
        from engine.galerkin.basis import TensorBasis
        c=config();c['dimension']=3
        c['grid']={'shape':[6,8,10],'lengths':[1.,1.2,1.4]}
        c['solver']['modes']=[3,3,3]
        c['initial_fields'][0]['initial']={'value':1.}
        other=deepcopy(c['initial_fields'][0]);other['id']='second_T';other['initial']={'value':2.}
        c['initial_fields'].append(other)
        c['events']=[{'type':'p_birth','time':.05,'source_T_id':'second_T'}]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m=SegmentedRun(c,root).run()
            self.assertEqual(len(m['segments']),2)
            child=m['field_registry'][-1]
            self.assertEqual(child['source_T_id'],'second_T')
            a,b=[read_archive(root/s['archive']) for s in m['segments']]
            basis=TensorBasis.from_descriptor(decode(a,'basis_json'))
            source=basis.reconstruct(a['coefficients'][-1,a['fields']=='second_T'].reshape(basis.shape))
            expected=basis.project((np.maximum(source,0)/(max(source.ravel())+1e-12))**3).ravel()
            np.testing.assert_allclose(b['coefficients'][0,b['fields']==child['id']],expected,atol=1e-12)

    def test_finite_volume_semantic_selector(self):
        from engine.simulation.solver import Simulation
        s=Simulation(config());s.add_field('P_existing','P',2.,0,{'diffusion':.001})
        before=s.u.copy()
        event=normalize_event({'type':'catastrophe','time':0,'strength':.5,
                               'target_selector':{'mode':'family','families':['T']}},1)
        s.apply(event)
        np.testing.assert_allclose(s.u[0],before[0]*.5)
        np.testing.assert_array_equal(s.u[1],before[1])
        self.assertEqual(event.parameters['resolved_target_ids'],['T_root'])
